"""SQLite market-data store for candles and auxiliary exchange metrics."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

from .api import Candle


@dataclass(frozen=True)
class CandleCoverage:
    symbol: str
    market_type: str
    interval: str
    count: int
    first_open_time: int | None
    last_open_time: int | None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandleCoverageQuality:
    coverage: CandleCoverage
    expected_count: int
    gap_count: int
    coverage_ratio: float

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["coverage"] = self.coverage.asdict()
        return payload


@dataclass(frozen=True)
class AggTrade:
    symbol: str
    market_type: str
    agg_trade_id: int
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    trade_time_ms: int
    is_buyer_maker: bool
    best_match: bool = True

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AggTradeCoverage:
    symbol: str
    market_type: str
    count: int
    first_trade_time_ms: int | None
    last_trade_time_ms: int | None
    first_agg_trade_id: int | None
    last_agg_trade_id: int | None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AggTradeBucket:
    symbol: str
    market_type: str
    open_time: int
    first_time_ms: int
    last_time_ms: int
    first_price: float
    last_price: float
    high_price: float
    low_price: float
    total_quantity: float
    total_notional: float
    buy_quantity: float
    buy_notional: float
    sell_quantity: float
    sell_notional: float
    aggregate_count: int
    buyer_taker_count: int
    seller_taker_count: int
    max_trade_notional: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TopOfBookSnapshot:
    symbol: str
    market_type: str
    provider: str
    ts_ms: int
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    mid_price: float
    spread: float
    spread_bps: float
    depth_notional: float
    ingested_at_ms: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveFileRecord:
    url: str
    symbol: str
    market_type: str
    interval: str
    period: str
    status: str
    rows_inserted: int
    bytes_downloaded: int
    sha256: str
    checksum_sha256: str
    checksum_status: str
    error: str
    started_at_ms: int
    completed_at_ms: int | None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FuturesReferenceBar:
    symbol: str
    market_type: str
    kind: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    close_time: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FundingRateRecord:
    symbol: str
    market_type: str
    calc_time: int
    funding_interval_hours: int
    funding_rate: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DerivativesArchiveFileRecord:
    url: str
    symbol: str
    market_type: str
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
    error: str
    started_at_ms: int
    completed_at_ms: int | None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class MarketDataStore:
    """Small SQLite store optimized for append/update market-data ingestion."""

    def __init__(self, path: str | Path = "data/market_data.sqlite") -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MarketDataStore":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        conn = self.connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                close_time INTEGER NOT NULL,
                quote_volume REAL NOT NULL DEFAULT 0,
                trade_count INTEGER NOT NULL DEFAULT 0,
                taker_buy_base_volume REAL NOT NULL DEFAULT 0,
                taker_buy_quote_volume REAL NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                ingested_at_ms INTEGER NOT NULL,
                PRIMARY KEY (symbol, market_type, interval, open_time)
            );
            CREATE INDEX IF NOT EXISTS idx_candles_lookup
                ON candles(symbol, market_type, interval, open_time);

            CREATE TABLE IF NOT EXISTS market_snapshots (
                provider TEXT NOT NULL,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                kind TEXT NOT NULL,
                ts_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (provider, symbol, market_type, kind, ts_ms)
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_lookup
                ON market_snapshots(symbol, market_type, kind, ts_ms);

            CREATE TABLE IF NOT EXISTS top_of_book_snapshots (
                provider TEXT NOT NULL,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                ts_ms INTEGER NOT NULL,
                bid_price REAL NOT NULL,
                bid_qty REAL NOT NULL,
                ask_price REAL NOT NULL,
                ask_qty REAL NOT NULL,
                mid_price REAL NOT NULL,
                spread REAL NOT NULL,
                spread_bps REAL NOT NULL,
                depth_notional REAL NOT NULL,
                payload_json TEXT NOT NULL,
                ingested_at_ms INTEGER NOT NULL,
                PRIMARY KEY (provider, symbol, market_type, ts_ms)
            );
            CREATE INDEX IF NOT EXISTS idx_top_of_book_lookup
                ON top_of_book_snapshots(symbol, market_type, ts_ms);

            CREATE TABLE IF NOT EXISTS agg_trades (
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                agg_trade_id INTEGER NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                first_trade_id INTEGER NOT NULL,
                last_trade_id INTEGER NOT NULL,
                trade_time_ms INTEGER NOT NULL,
                is_buyer_maker INTEGER NOT NULL,
                best_match INTEGER NOT NULL,
                source TEXT NOT NULL,
                ingested_at_ms INTEGER NOT NULL,
                PRIMARY KEY (symbol, market_type, agg_trade_id)
            );
            CREATE INDEX IF NOT EXISTS idx_agg_trades_time_lookup
                ON agg_trades(symbol, market_type, trade_time_ms);
            CREATE INDEX IF NOT EXISTS idx_agg_trades_id_lookup
                ON agg_trades(symbol, market_type, agg_trade_id);

            CREATE TABLE IF NOT EXISTS sync_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_rate_limit_snapshots (
                provider TEXT NOT NULL,
                market_type TEXT NOT NULL,
                ts_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (provider, market_type, ts_ms)
            );
            CREATE INDEX IF NOT EXISTS idx_api_rate_limit_latest
                ON api_rate_limit_snapshots(provider, market_type, ts_ms);

            CREATE TABLE IF NOT EXISTS archive_files (
                url TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                interval TEXT NOT NULL,
                period TEXT NOT NULL,
                status TEXT NOT NULL,
                rows_inserted INTEGER NOT NULL DEFAULT 0,
                bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                sha256 TEXT NOT NULL DEFAULT '',
                checksum_sha256 TEXT NOT NULL DEFAULT '',
                checksum_status TEXT NOT NULL DEFAULT 'unverified',
                error TEXT NOT NULL DEFAULT '',
                started_at_ms INTEGER NOT NULL,
                completed_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_archive_files_lookup
                ON archive_files(symbol, market_type, interval, status);

            CREATE TABLE IF NOT EXISTS derivatives_archive_files (
                url TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                data_type TEXT NOT NULL,
                interval TEXT NOT NULL,
                period TEXT NOT NULL,
                status TEXT NOT NULL,
                rows_inserted INTEGER NOT NULL DEFAULT 0,
                rows_read INTEGER NOT NULL DEFAULT 0,
                bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                sha256 TEXT NOT NULL DEFAULT '',
                checksum_sha256 TEXT NOT NULL DEFAULT '',
                checksum_status TEXT NOT NULL DEFAULT 'unverified',
                row_stream_sha256 TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                started_at_ms INTEGER NOT NULL,
                completed_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_derivatives_archive_files_lookup
                ON derivatives_archive_files(
                    symbol, market_type, data_type, interval, status, period
                );

            CREATE TABLE IF NOT EXISTS futures_reference_bars (
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                kind TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                close_time INTEGER NOT NULL,
                source TEXT NOT NULL,
                ingested_at_ms INTEGER NOT NULL,
                PRIMARY KEY (symbol, market_type, kind, interval, open_time)
            );
            CREATE INDEX IF NOT EXISTS idx_futures_reference_bars_lookup
                ON futures_reference_bars(
                    symbol, market_type, kind, interval, open_time
                );

            CREATE TABLE IF NOT EXISTS funding_rates (
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                calc_time INTEGER NOT NULL,
                funding_interval_hours INTEGER NOT NULL,
                funding_rate REAL NOT NULL,
                source TEXT NOT NULL,
                ingested_at_ms INTEGER NOT NULL,
                PRIMARY KEY (symbol, market_type, calc_time)
            );
            CREATE INDEX IF NOT EXISTS idx_funding_rates_lookup
                ON funding_rates(symbol, market_type, calc_time);

            CREATE TABLE IF NOT EXISTS microstructure_captures (
                capture_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                market_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at_ms INTEGER NOT NULL,
                completed_at_ms INTEGER NOT NULL,
                output_dir TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_microstructure_captures_status
                ON microstructure_captures(status, completed_at_ms);

            CREATE TABLE IF NOT EXISTS microstructure_capture_symbols (
                capture_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                normalized_path TEXT NOT NULL,
                initial_snapshot_path TEXT NOT NULL,
                raw_sha256 TEXT NOT NULL,
                normalized_sha256 TEXT NOT NULL,
                raw_messages INTEGER NOT NULL,
                normalized_rows INTEGER NOT NULL,
                depth_messages INTEGER NOT NULL,
                trade_messages INTEGER NOT NULL,
                book_ticker_messages INTEGER NOT NULL,
                sequence_gap_count INTEGER NOT NULL,
                crossed_book_count INTEGER NOT NULL,
                invalid_event_count INTEGER NOT NULL,
                replay_smoke_passed INTEGER NOT NULL,
                error TEXT NOT NULL,
                PRIMARY KEY (capture_id, symbol),
                FOREIGN KEY (capture_id) REFERENCES microstructure_captures(capture_id)
            );
            CREATE INDEX IF NOT EXISTS idx_microstructure_symbols_lookup
                ON microstructure_capture_symbols(symbol, capture_id);
            """
        )
        self._ensure_archive_file_columns(conn)
        conn.commit()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_archive_file_columns(self, conn: sqlite3.Connection) -> None:
        columns = self._table_columns(conn, "archive_files")
        if "checksum_sha256" not in columns:
            conn.execute("ALTER TABLE archive_files ADD COLUMN checksum_sha256 TEXT NOT NULL DEFAULT ''")
        if "checksum_status" not in columns:
            conn.execute("ALTER TABLE archive_files ADD COLUMN checksum_status TEXT NOT NULL DEFAULT 'unverified'")

    def record_microstructure_capture(self, payload: Mapping[str, object]) -> int:
        """Persist one immutable L2 capture manifest and its per-symbol evidence."""

        capture_id = str(payload.get("capture_id", "")).strip()
        evidence = payload.get("evidence")
        if not capture_id or not isinstance(evidence, list):
            raise ValueError("microstructure capture requires capture_id and evidence")
        encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        conn = self.connect()
        existing = conn.execute(
            "SELECT payload_json FROM microstructure_captures WHERE capture_id = ?",
            (capture_id,),
        ).fetchone()
        if existing is not None:
            if str(existing["payload_json"]) == encoded:
                return 0
            raise ValueError(f"microstructure capture_id is immutable: {capture_id}")
        before_changes = conn.total_changes
        with conn:
            conn.execute(
                """
                INSERT INTO microstructure_captures (
                    capture_id, provider, market_type, schema_version, status,
                    started_at_ms, completed_at_ms, output_dir, manifest_path, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_id,
                    str(payload.get("provider", "")),
                    str(payload.get("market_type", "")),
                    str(payload.get("schema_version", "")),
                    str(payload.get("status", "")),
                    int(payload.get("started_at_ms", 0)),
                    int(payload.get("completed_at_ms", 0)),
                    str(payload.get("output_dir", "")),
                    str(payload.get("manifest_path", "")),
                    encoded,
                ),
            )
            conn.execute(
                "DELETE FROM microstructure_capture_symbols WHERE capture_id = ?",
                (capture_id,),
            )
            rows: list[tuple[object, ...]] = []
            for item in evidence:
                if not isinstance(item, Mapping):
                    raise ValueError("microstructure symbol evidence must be an object")
                symbol = str(item.get("symbol", "")).strip().upper()
                if not symbol:
                    raise ValueError("microstructure symbol evidence requires symbol")
                rows.append(
                    (
                        capture_id,
                        symbol,
                        str(item.get("raw_path", "")),
                        str(item.get("normalized_path", "")),
                        str(item.get("initial_snapshot_path", "")),
                        str(item.get("raw_sha256", "")),
                        str(item.get("normalized_sha256", "")),
                        int(item.get("raw_messages", 0)),
                        int(item.get("normalized_rows", 0)),
                        int(item.get("depth_messages", 0)),
                        int(item.get("trade_messages", 0)),
                        int(item.get("book_ticker_messages", 0)),
                        int(item.get("sequence_gap_count", 0)),
                        int(item.get("crossed_book_count", 0)),
                        int(item.get("invalid_event_count", 0)),
                        1 if bool(item.get("replay_smoke_passed", False)) else 0,
                        str(item.get("error", "")),
                    )
                )
            conn.executemany(
                """
                INSERT INTO microstructure_capture_symbols (
                    capture_id, symbol, raw_path, normalized_path, initial_snapshot_path,
                    raw_sha256, normalized_sha256, raw_messages, normalized_rows,
                    depth_messages, trade_messages, book_ticker_messages,
                    sequence_gap_count, crossed_book_count, invalid_event_count,
                    replay_smoke_passed, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return conn.total_changes - before_changes

    def latest_microstructure_capture(
        self,
        symbol: str,
        *,
        require_passed: bool = True,
    ) -> dict[str, object] | None:
        """Return the newest catalogued capture containing one symbol."""

        conditions = ["s.symbol = ?"]
        parameters: list[object] = [str(symbol).upper()]
        if require_passed:
            conditions.extend(
                (
                    "c.status = 'pass'",
                    "s.replay_smoke_passed = 1",
                    "s.sequence_gap_count = 0",
                    "s.crossed_book_count = 0",
                    "s.invalid_event_count = 0",
                    "s.error = ''",
                )
            )
        row = self.connect().execute(
            f"""
            SELECT c.payload_json
            FROM microstructure_captures c
            JOIN microstructure_capture_symbols s ON s.capture_id = c.capture_id
            WHERE {' AND '.join(conditions)}
            ORDER BY c.completed_at_ms DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return cast(dict[str, object], payload)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def upsert_candles(
        self,
        symbol: str,
        market_type: str,
        interval: str,
        candles: Sequence[Candle],
        *,
        source: str = "binance",
        ingested_at_ms: int | None = None,
    ) -> int:
        if not candles:
            return 0
        ingested = self._now_ms() if ingested_at_ms is None else int(ingested_at_ms)
        rows = [
            (
                symbol.upper(),
                market_type,
                interval,
                int(candle.open_time),
                float(candle.open),
                float(candle.high),
                float(candle.low),
                float(candle.close),
                float(candle.volume),
                int(candle.close_time),
                float(candle.quote_volume),
                int(candle.trade_count),
                float(candle.taker_buy_base_volume),
                float(candle.taker_buy_quote_volume),
                source,
                ingested,
            )
            for candle in candles
        ]
        conn = self.connect()
        before_changes = conn.total_changes
        conn.executemany(
            """
            INSERT INTO candles (
                symbol, market_type, interval, open_time, open, high, low, close, volume,
                close_time, quote_volume, trade_count, taker_buy_base_volume,
                taker_buy_quote_volume, source, ingested_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, market_type, interval, open_time) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                close_time=excluded.close_time,
                quote_volume=excluded.quote_volume,
                trade_count=excluded.trade_count,
                taker_buy_base_volume=excluded.taker_buy_base_volume,
                taker_buy_quote_volume=excluded.taker_buy_quote_volume,
                source=excluded.source,
                ingested_at_ms=excluded.ingested_at_ms
            WHERE
                candles.open IS NOT excluded.open OR
                candles.high IS NOT excluded.high OR
                candles.low IS NOT excluded.low OR
                candles.close IS NOT excluded.close OR
                candles.volume IS NOT excluded.volume OR
                candles.close_time IS NOT excluded.close_time OR
                candles.quote_volume IS NOT excluded.quote_volume OR
                candles.trade_count IS NOT excluded.trade_count OR
                candles.taker_buy_base_volume IS NOT excluded.taker_buy_base_volume OR
                candles.taker_buy_quote_volume IS NOT excluded.taker_buy_quote_volume OR
                candles.source IS NOT excluded.source
            """,
            rows,
        )
        conn.commit()
        return max(0, conn.total_changes - before_changes)

    def fetch_candles(
        self,
        symbol: str,
        market_type: str,
        interval: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        params: list[object] = [symbol.upper(), market_type, interval]
        where = ["symbol = ?", "market_type = ?", "interval = ?"]
        if start_ms is not None:
            where.append("open_time >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("open_time <= ?")
            params.append(int(end_ms))
        query = """
            SELECT open_time, open, high, low, close, volume, close_time,
                   quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume
            FROM candles
            WHERE """ + " AND ".join(where) + """
            ORDER BY open_time DESC
            """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.connect().execute(query, params).fetchall()
        return [
            Candle(
                open_time=int(row["open_time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                close_time=int(row["close_time"]),
                quote_volume=float(row["quote_volume"]),
                trade_count=int(row["trade_count"]),
                taker_buy_base_volume=float(row["taker_buy_base_volume"]),
                taker_buy_quote_volume=float(row["taker_buy_quote_volume"]),
            )
            for row in reversed(rows)
        ]

    def coverage(
        self,
        symbol: str,
        market_type: str,
        interval: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> CandleCoverage:
        params: list[object] = [symbol.upper(), market_type, interval]
        where = ["symbol = ?", "market_type = ?", "interval = ?"]
        if start_ms is not None:
            where.append("open_time >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("open_time <= ?")
            params.append(int(end_ms))
        row = self.connect().execute(
            f"""
            SELECT COUNT(*) AS count, MIN(open_time) AS first_open_time, MAX(open_time) AS last_open_time
            FROM candles
            WHERE {' AND '.join(where)}
            """,
            params,
        ).fetchone()
        return CandleCoverage(
            symbol=symbol.upper(),
            market_type=market_type,
            interval=interval,
            count=int(row["count"]),
            first_open_time=row["first_open_time"],
            last_open_time=row["last_open_time"],
        )

    def candle_series(
        self,
        *,
        market_type: str | None = None,
        interval: str | None = None,
    ) -> list[CandleCoverage]:
        where: list[str] = []
        params: list[object] = []
        if market_type is not None:
            where.append("market_type = ?")
            params.append(market_type)
        if interval is not None:
            where.append("interval = ?")
            params.append(interval)
        query = """
            SELECT symbol, market_type, interval, COUNT(*) AS count,
                   MIN(open_time) AS first_open_time, MAX(open_time) AS last_open_time
            FROM candles
            """
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " GROUP BY symbol, market_type, interval ORDER BY symbol, market_type, interval"
        rows = self.connect().execute(query, params).fetchall()
        return [
            CandleCoverage(
                symbol=str(row["symbol"]),
                market_type=str(row["market_type"]),
                interval=str(row["interval"]),
                count=int(row["count"]),
                first_open_time=row["first_open_time"],
                last_open_time=row["last_open_time"],
            )
            for row in rows
        ]

    def coverage_quality(
        self,
        symbol: str,
        market_type: str,
        interval: str,
        interval_ms: int,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> CandleCoverageQuality:
        coverage = self.coverage(symbol, market_type, interval, start_ms=start_ms, end_ms=end_ms)
        if coverage.count == 0:
            if interval_ms > 0 and start_ms is not None and end_ms is not None and int(end_ms) >= int(start_ms):
                expected_count = ((int(end_ms) - int(start_ms)) // interval_ms) + 1
                return CandleCoverageQuality(
                    coverage,
                    expected_count=expected_count,
                    gap_count=expected_count,
                    coverage_ratio=0.0,
                )
            return CandleCoverageQuality(coverage, expected_count=0, gap_count=0, coverage_ratio=0.0)
        if interval_ms <= 0:
            return CandleCoverageQuality(
                coverage,
                expected_count=coverage.count,
                gap_count=0,
                coverage_ratio=1.0,
            )

        params: list[object] = [symbol.upper(), market_type, interval]
        where = ["symbol = ?", "market_type = ?", "interval = ?"]
        if start_ms is not None:
            where.append("open_time >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("open_time <= ?")
            params.append(int(end_ms))
        rows = self.connect().execute(
            f"""
            SELECT open_time
            FROM candles
            WHERE {' AND '.join(where)}
            ORDER BY open_time ASC
            """,
            params,
        )
        first_open_time = cast(int, coverage.first_open_time)
        last_open_time = cast(int, coverage.last_open_time)
        span_start = int(start_ms) if start_ms is not None else first_open_time
        span_end = int(end_ms) if end_ms is not None else last_open_time
        missing = 0
        if first_open_time > span_start:
            missing += max(0, (first_open_time - span_start) // interval_ms)
        previous: int | None = None
        for row in rows:
            if row["open_time"] is None:
                continue
            current = int(row["open_time"])
            if previous is not None:
                delta = current - previous
                if delta > interval_ms:
                    missing += max(0, (delta // interval_ms) - 1)
            previous = current
        if last_open_time < span_end:
            missing += max(0, (span_end - last_open_time) // interval_ms)

        span = max(0, span_end - span_start)
        expected_count = max(coverage.count, (span // interval_ms) + 1)
        ratio = coverage.count / expected_count if expected_count else 0.0
        return CandleCoverageQuality(
            coverage,
            expected_count=expected_count,
            gap_count=missing,
            coverage_ratio=ratio,
        )

    def latest_open_time(self, symbol: str, market_type: str, interval: str) -> int | None:
        return self.coverage(symbol, market_type, interval).last_open_time

    def upsert_agg_trades(
        self,
        symbol: str,
        market_type: str,
        trades: Sequence[AggTrade],
        *,
        source: str = "binance_public_archive_aggTrades",
        ingested_at_ms: int | None = None,
    ) -> int:
        if not trades:
            return 0
        normalized_symbol = symbol.upper()
        ingested = self._now_ms() if ingested_at_ms is None else int(ingested_at_ms)
        rows: list[tuple[object, ...]] = []
        for trade in trades:
            price = float(trade.price)
            quantity = float(trade.quantity)
            trade_time_ms = int(trade.trade_time_ms)
            if price <= 0.0 or quantity <= 0.0 or trade_time_ms <= 0:
                raise ValueError("aggregate trade rows require positive price, quantity, and trade_time_ms")
            first_trade_id = int(trade.first_trade_id)
            last_trade_id = int(trade.last_trade_id)
            if last_trade_id < first_trade_id:
                raise ValueError("aggregate trade last_trade_id must be greater than or equal to first_trade_id")
            rows.append(
                (
                    normalized_symbol,
                    market_type,
                    int(trade.agg_trade_id),
                    price,
                    quantity,
                    first_trade_id,
                    last_trade_id,
                    trade_time_ms,
                    1 if bool(trade.is_buyer_maker) else 0,
                    1 if bool(trade.best_match) else 0,
                    source,
                    ingested,
                )
            )
        conn = self.connect()
        before_changes = conn.total_changes
        conn.executemany(
            """
            INSERT INTO agg_trades (
                symbol, market_type, agg_trade_id, price, quantity, first_trade_id, last_trade_id,
                trade_time_ms, is_buyer_maker, best_match, source, ingested_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, market_type, agg_trade_id) DO UPDATE SET
                price=excluded.price,
                quantity=excluded.quantity,
                first_trade_id=excluded.first_trade_id,
                last_trade_id=excluded.last_trade_id,
                trade_time_ms=excluded.trade_time_ms,
                is_buyer_maker=excluded.is_buyer_maker,
                best_match=excluded.best_match,
                source=excluded.source,
                ingested_at_ms=excluded.ingested_at_ms
            WHERE
                agg_trades.price IS NOT excluded.price OR
                agg_trades.quantity IS NOT excluded.quantity OR
                agg_trades.first_trade_id IS NOT excluded.first_trade_id OR
                agg_trades.last_trade_id IS NOT excluded.last_trade_id OR
                agg_trades.trade_time_ms IS NOT excluded.trade_time_ms OR
                agg_trades.is_buyer_maker IS NOT excluded.is_buyer_maker OR
                agg_trades.best_match IS NOT excluded.best_match OR
                agg_trades.source IS NOT excluded.source
            """,
            rows,
        )
        conn.commit()
        return max(0, conn.total_changes - before_changes)

    @staticmethod
    def _agg_trade_from_row(row: sqlite3.Row) -> AggTrade:
        return AggTrade(
            symbol=str(row["symbol"]),
            market_type=str(row["market_type"]),
            agg_trade_id=int(row["agg_trade_id"]),
            price=float(row["price"]),
            quantity=float(row["quantity"]),
            first_trade_id=int(row["first_trade_id"]),
            last_trade_id=int(row["last_trade_id"]),
            trade_time_ms=int(row["trade_time_ms"]),
            is_buyer_maker=bool(int(row["is_buyer_maker"])),
            best_match=bool(int(row["best_match"])),
        )

    def fetch_agg_trades(
        self,
        symbol: str,
        market_type: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[AggTrade]:
        params: list[object] = [symbol.upper(), market_type]
        where = ["symbol = ?", "market_type = ?"]
        if start_ms is not None:
            where.append("trade_time_ms >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("trade_time_ms <= ?")
            params.append(int(end_ms))
        query = f"""
            SELECT symbol, market_type, agg_trade_id, price, quantity, first_trade_id, last_trade_id,
                   trade_time_ms, is_buyer_maker, best_match
            FROM agg_trades
            WHERE {' AND '.join(where)}
            ORDER BY trade_time_ms DESC, agg_trade_id DESC
            """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.connect().execute(query, params).fetchall()
        return [self._agg_trade_from_row(row) for row in reversed(rows)]

    def agg_trade_coverage(
        self,
        symbol: str,
        market_type: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> AggTradeCoverage:
        params: list[object] = [symbol.upper(), market_type]
        where = ["symbol = ?", "market_type = ?"]
        if start_ms is not None:
            where.append("trade_time_ms >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("trade_time_ms <= ?")
            params.append(int(end_ms))
        row = self.connect().execute(
            f"""
            SELECT COUNT(*) AS count,
                   MIN(trade_time_ms) AS first_trade_time_ms,
                   MAX(trade_time_ms) AS last_trade_time_ms,
                   MIN(agg_trade_id) AS first_agg_trade_id,
                   MAX(agg_trade_id) AS last_agg_trade_id
            FROM agg_trades
            WHERE {' AND '.join(where)}
            """,
            params,
        ).fetchone()
        return AggTradeCoverage(
            symbol=symbol.upper(),
            market_type=market_type,
            count=int(row["count"]),
            first_trade_time_ms=row["first_trade_time_ms"],
            last_trade_time_ms=row["last_trade_time_ms"],
            first_agg_trade_id=row["first_agg_trade_id"],
            last_agg_trade_id=row["last_agg_trade_id"],
        )

    def fetch_agg_trade_buckets(
        self,
        symbol: str,
        market_type: str,
        *,
        bucket_ms: int = 1000,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[AggTradeBucket]:
        width = max(1, int(bucket_ms))
        params: list[object] = [width, width, symbol.upper(), market_type]
        where = ["symbol = ?", "market_type = ?"]
        if start_ms is not None:
            where.append("trade_time_ms >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("trade_time_ms <= ?")
            params.append(int(end_ms))
        query = f"""
            WITH normalized AS (
                SELECT
                    ((trade_time_ms / ?) * ?) AS bucket_open_time,
                    symbol,
                    market_type,
                    agg_trade_id,
                    price,
                    quantity,
                    price * quantity AS notional,
                    trade_time_ms,
                    is_buyer_maker
                FROM agg_trades
                WHERE {' AND '.join(where)}
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY bucket_open_time
                        ORDER BY trade_time_ms ASC, agg_trade_id ASC
                    ) AS rn_first,
                    ROW_NUMBER() OVER (
                        PARTITION BY bucket_open_time
                        ORDER BY trade_time_ms DESC, agg_trade_id DESC
                    ) AS rn_last
                FROM normalized
            )
            SELECT
                bucket_open_time,
                MIN(symbol) AS symbol,
                MIN(market_type) AS market_type,
                MIN(trade_time_ms) AS first_time_ms,
                MAX(trade_time_ms) AS last_time_ms,
                MAX(CASE WHEN rn_first = 1 THEN price END) AS first_price,
                MAX(CASE WHEN rn_last = 1 THEN price END) AS last_price,
                MAX(price) AS high_price,
                MIN(price) AS low_price,
                SUM(quantity) AS total_quantity,
                SUM(notional) AS total_notional,
                SUM(CASE WHEN is_buyer_maker = 0 THEN quantity ELSE 0 END) AS buy_quantity,
                SUM(CASE WHEN is_buyer_maker = 0 THEN notional ELSE 0 END) AS buy_notional,
                SUM(CASE WHEN is_buyer_maker != 0 THEN quantity ELSE 0 END) AS sell_quantity,
                SUM(CASE WHEN is_buyer_maker != 0 THEN notional ELSE 0 END) AS sell_notional,
                COUNT(*) AS aggregate_count,
                SUM(CASE WHEN is_buyer_maker = 0 THEN 1 ELSE 0 END) AS buyer_taker_count,
                SUM(CASE WHEN is_buyer_maker != 0 THEN 1 ELSE 0 END) AS seller_taker_count,
                MAX(notional) AS max_trade_notional
            FROM ranked
            GROUP BY bucket_open_time
            ORDER BY bucket_open_time DESC
            """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.connect().execute(query, params).fetchall()
        return [
            AggTradeBucket(
                symbol=str(row["symbol"]),
                market_type=str(row["market_type"]),
                open_time=int(row["bucket_open_time"]),
                first_time_ms=int(row["first_time_ms"]),
                last_time_ms=int(row["last_time_ms"]),
                first_price=float(row["first_price"]),
                last_price=float(row["last_price"]),
                high_price=float(row["high_price"]),
                low_price=float(row["low_price"]),
                total_quantity=float(row["total_quantity"]),
                total_notional=float(row["total_notional"]),
                buy_quantity=float(row["buy_quantity"]),
                buy_notional=float(row["buy_notional"]),
                sell_quantity=float(row["sell_quantity"]),
                sell_notional=float(row["sell_notional"]),
                aggregate_count=int(row["aggregate_count"]),
                buyer_taker_count=int(row["buyer_taker_count"]),
                seller_taker_count=int(row["seller_taker_count"]),
                max_trade_notional=float(row["max_trade_notional"]),
            )
            for row in reversed(rows)
        ]

    @staticmethod
    def _finite_positive(payload: Mapping[str, object], key: str) -> float:
        try:
            value = float(payload[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"top-of-book missing numeric {key}") from exc
        if not value > 0.0:
            raise ValueError(f"top-of-book {key} must be positive")
        return value

    def insert_top_of_book_snapshot(
        self,
        provider: str,
        symbol: str,
        market_type: str,
        payload: Mapping[str, object],
        *,
        ts_ms: int | None = None,
        ingested_at_ms: int | None = None,
    ) -> int:
        timestamp = self._now_ms() if ts_ms is None else int(ts_ms)
        ingested = self._now_ms() if ingested_at_ms is None else int(ingested_at_ms)
        bid_price = self._finite_positive(payload, "bidPrice")
        ask_price = self._finite_positive(payload, "askPrice")
        bid_qty = self._finite_positive(payload, "bidQty")
        ask_qty = self._finite_positive(payload, "askQty")
        if ask_price < bid_price:
            raise ValueError("top-of-book askPrice is below bidPrice")
        mid_price = (bid_price + ask_price) / 2.0
        spread = ask_price - bid_price
        spread_bps = (spread / mid_price) * 10_000.0 if mid_price > 0.0 else 0.0
        depth_notional = bid_price * bid_qty + ask_price * ask_qty
        before_changes = self.connect().total_changes
        self.connect().execute(
            """
            INSERT INTO top_of_book_snapshots (
                provider, symbol, market_type, ts_ms, bid_price, bid_qty, ask_price, ask_qty,
                mid_price, spread, spread_bps, depth_notional, payload_json, ingested_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, symbol, market_type, ts_ms) DO UPDATE SET
                bid_price=excluded.bid_price,
                bid_qty=excluded.bid_qty,
                ask_price=excluded.ask_price,
                ask_qty=excluded.ask_qty,
                mid_price=excluded.mid_price,
                spread=excluded.spread,
                spread_bps=excluded.spread_bps,
                depth_notional=excluded.depth_notional,
                payload_json=excluded.payload_json,
                ingested_at_ms=excluded.ingested_at_ms
            WHERE
                top_of_book_snapshots.bid_price IS NOT excluded.bid_price OR
                top_of_book_snapshots.bid_qty IS NOT excluded.bid_qty OR
                top_of_book_snapshots.ask_price IS NOT excluded.ask_price OR
                top_of_book_snapshots.ask_qty IS NOT excluded.ask_qty OR
                top_of_book_snapshots.payload_json IS NOT excluded.payload_json
            """,
            (
                provider,
                symbol.upper(),
                market_type,
                timestamp,
                bid_price,
                bid_qty,
                ask_price,
                ask_qty,
                mid_price,
                spread,
                spread_bps,
                depth_notional,
                json.dumps(dict(payload), sort_keys=True),
                ingested,
            ),
        )
        self.connect().commit()
        return max(0, self.connect().total_changes - before_changes)

    @staticmethod
    def _top_of_book_from_row(row: sqlite3.Row) -> TopOfBookSnapshot:
        return TopOfBookSnapshot(
            symbol=str(row["symbol"]),
            market_type=str(row["market_type"]),
            provider=str(row["provider"]),
            ts_ms=int(row["ts_ms"]),
            bid_price=float(row["bid_price"]),
            bid_qty=float(row["bid_qty"]),
            ask_price=float(row["ask_price"]),
            ask_qty=float(row["ask_qty"]),
            mid_price=float(row["mid_price"]),
            spread=float(row["spread"]),
            spread_bps=float(row["spread_bps"]),
            depth_notional=float(row["depth_notional"]),
            ingested_at_ms=int(row["ingested_at_ms"]),
        )

    def fetch_top_of_book(
        self,
        symbol: str,
        market_type: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> list[TopOfBookSnapshot]:
        params: list[object] = [symbol.upper(), market_type]
        where = ["symbol = ?", "market_type = ?"]
        if start_ms is not None:
            where.append("ts_ms >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("ts_ms <= ?")
            params.append(int(end_ms))
        query = f"""
            SELECT provider, symbol, market_type, ts_ms, bid_price, bid_qty, ask_price, ask_qty,
                   mid_price, spread, spread_bps, depth_notional, ingested_at_ms
            FROM top_of_book_snapshots
            WHERE {' AND '.join(where)}
            ORDER BY ts_ms DESC
            """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self.connect().execute(query, params).fetchall()
        return [self._top_of_book_from_row(row) for row in reversed(rows)]

    def latest_top_of_book(self, symbol: str, market_type: str) -> TopOfBookSnapshot | None:
        rows = self.fetch_top_of_book(symbol, market_type, limit=1)
        return rows[-1] if rows else None

    def insert_snapshot(
        self,
        provider: str,
        symbol: str,
        market_type: str,
        kind: str,
        payload: Mapping[str, object] | Sequence[Mapping[str, object]],
        *,
        ts_ms: int | None = None,
    ) -> int:
        timestamp = self._now_ms() if ts_ms is None else int(ts_ms)
        self.connect().execute(
            """
            INSERT OR REPLACE INTO market_snapshots
                (provider, symbol, market_type, kind, ts_ms, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                provider,
                symbol.upper(),
                market_type,
                kind,
                timestamp,
                json.dumps(payload, sort_keys=True),
            ),
        )
        self.connect().commit()
        return 1

    def latest_snapshot(self, symbol: str, market_type: str, kind: str) -> dict[str, object] | list[object] | None:
        row = self.connect().execute(
            """
            SELECT payload_json
            FROM market_snapshots
            WHERE symbol = ? AND market_type = ? AND kind = ?
            ORDER BY ts_ms DESC
            LIMIT 1
            """,
            (symbol.upper(), market_type, kind),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return payload if isinstance(payload, (dict, list)) else None

    def insert_sync_run(self, payload: Mapping[str, object]) -> int:
        cursor = self.connect().execute(
            "INSERT INTO sync_runs (created_at_ms, payload_json) VALUES (?, ?)",
            (self._now_ms(), json.dumps(dict(payload), sort_keys=True)),
        )
        self.connect().commit()
        return int(cast(int, cursor.lastrowid))

    def insert_api_rate_limit_snapshot(
        self,
        provider: str,
        market_type: str,
        payload: Mapping[str, object],
        *,
        ts_ms: int | None = None,
    ) -> int:
        timestamp = self._now_ms() if ts_ms is None else int(ts_ms)
        before_changes = self.connect().total_changes
        self.connect().execute(
            """
            INSERT INTO api_rate_limit_snapshots(provider, market_type, ts_ms, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider, market_type, ts_ms) DO UPDATE SET
                payload_json=excluded.payload_json
            WHERE api_rate_limit_snapshots.payload_json IS NOT excluded.payload_json
            """,
            (provider, market_type, timestamp, json.dumps(dict(payload), sort_keys=True)),
        )
        self.connect().commit()
        return max(0, self.connect().total_changes - before_changes)

    def latest_api_rate_limit_snapshot(
        self,
        provider: str = "binance",
        market_type: str = "spot",
    ) -> dict[str, object] | None:
        row = self.connect().execute(
            """
            SELECT payload_json
            FROM api_rate_limit_snapshots
            WHERE provider = ? AND market_type = ?
            ORDER BY ts_ms DESC
            LIMIT 1
            """,
            (provider, market_type),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return payload if isinstance(payload, dict) else None

    def begin_archive_file(
        self,
        *,
        url: str,
        symbol: str,
        market_type: str,
        interval: str,
        period: str,
        started_at_ms: int | None = None,
    ) -> None:
        timestamp = self._now_ms() if started_at_ms is None else int(started_at_ms)
        self.connect().execute(
            """
            INSERT INTO archive_files(
                url, symbol, market_type, interval, period, status, rows_inserted,
                bytes_downloaded, sha256, checksum_sha256, checksum_status, error,
                started_at_ms, completed_at_ms
            )
            VALUES (?, ?, ?, ?, ?, 'started', 0, 0, '', '', 'unverified', '', ?, NULL)
            ON CONFLICT(url) DO UPDATE SET
                status='started',
                error='',
                started_at_ms=excluded.started_at_ms,
                completed_at_ms=NULL
            """,
            (url, symbol.upper(), market_type, interval, period, timestamp),
        )
        self.connect().commit()

    def complete_archive_file(
        self,
        *,
        url: str,
        status: str,
        rows_inserted: int,
        bytes_downloaded: int,
        sha256: str,
        checksum_sha256: str = "",
        checksum_status: str = "unverified",
        error: str = "",
        completed_at_ms: int | None = None,
    ) -> None:
        timestamp = self._now_ms() if completed_at_ms is None else int(completed_at_ms)
        self.connect().execute(
            """
            UPDATE archive_files
            SET status = ?,
                rows_inserted = ?,
                bytes_downloaded = ?,
                sha256 = ?,
                checksum_sha256 = ?,
                checksum_status = ?,
                error = ?,
                completed_at_ms = ?
            WHERE url = ?
            """,
            (
                status,
                max(0, int(rows_inserted)),
                max(0, int(bytes_downloaded)),
                str(sha256 or ""),
                str(checksum_sha256 or ""),
                str(checksum_status or "unverified"),
                str(error or ""),
                timestamp,
                url,
            ),
        )
        self.connect().commit()

    def archive_file_status(self, url: str) -> str | None:
        row = self.connect().execute(
            "SELECT status FROM archive_files WHERE url = ?",
            (url,),
        ).fetchone()
        return str(row["status"]) if row is not None else None

    def archive_files(
        self,
        *,
        symbol: str | None = None,
        market_type: str | None = None,
        interval: str | None = None,
        status: str | None = None,
    ) -> list[ArchiveFileRecord]:
        where: list[str] = []
        params: list[object] = []
        if symbol is not None:
            where.append("symbol = ?")
            params.append(symbol.upper())
        if market_type is not None:
            where.append("market_type = ?")
            params.append(market_type)
        if interval is not None:
            where.append("interval = ?")
            params.append(interval)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        query = """
            SELECT url, symbol, market_type, interval, period, status, rows_inserted,
                   bytes_downloaded, sha256, checksum_sha256, checksum_status,
                   error, started_at_ms, completed_at_ms
            FROM archive_files
            """
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY symbol, interval, url"
        rows = self.connect().execute(query, params).fetchall()
        return [
            ArchiveFileRecord(
                url=str(row["url"]),
                symbol=str(row["symbol"]),
                market_type=str(row["market_type"]),
                interval=str(row["interval"]),
                period=str(row["period"]),
                status=str(row["status"]),
                rows_inserted=int(row["rows_inserted"]),
                bytes_downloaded=int(row["bytes_downloaded"]),
                sha256=str(row["sha256"]),
                checksum_sha256=str(row["checksum_sha256"]),
                checksum_status=str(row["checksum_status"]),
                error=str(row["error"]),
                started_at_ms=int(row["started_at_ms"]),
                completed_at_ms=(int(row["completed_at_ms"]) if row["completed_at_ms"] is not None else None),
            )
            for row in rows
        ]

    def upsert_futures_reference_bars(
        self,
        records: Sequence[FuturesReferenceBar],
        *,
        source: str,
        ingested_at_ms: int | None = None,
    ) -> int:
        if not records:
            return 0
        timestamp = self._now_ms() if ingested_at_ms is None else int(ingested_at_ms)
        rows = [
            (
                record.symbol.upper(),
                record.market_type,
                record.kind,
                record.interval,
                int(record.open_time),
                float(record.open),
                float(record.high),
                float(record.low),
                float(record.close),
                int(record.close_time),
                source,
                timestamp,
            )
            for record in records
        ]
        before = self.connect().total_changes
        with self.connect():
            self.connect().executemany(
                """
            INSERT INTO futures_reference_bars(
                symbol, market_type, kind, interval, open_time, open, high, low,
                close, close_time, source, ingested_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, market_type, kind, interval, open_time) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                close_time=excluded.close_time,
                source=excluded.source,
                ingested_at_ms=excluded.ingested_at_ms
            WHERE futures_reference_bars.open IS NOT excluded.open
               OR futures_reference_bars.high IS NOT excluded.high
               OR futures_reference_bars.low IS NOT excluded.low
               OR futures_reference_bars.close IS NOT excluded.close
               OR futures_reference_bars.close_time IS NOT excluded.close_time
               OR futures_reference_bars.source IS NOT excluded.source
                """,
                rows,
            )
        return max(0, self.connect().total_changes - before)

    def upsert_funding_rates(
        self,
        records: Sequence[FundingRateRecord],
        *,
        source: str,
        ingested_at_ms: int | None = None,
    ) -> int:
        if not records:
            return 0
        timestamp = self._now_ms() if ingested_at_ms is None else int(ingested_at_ms)
        rows = [
            (
                record.symbol.upper(),
                record.market_type,
                int(record.calc_time),
                int(record.funding_interval_hours),
                float(record.funding_rate),
                source,
                timestamp,
            )
            for record in records
        ]
        before = self.connect().total_changes
        with self.connect():
            self.connect().executemany(
                """
            INSERT INTO funding_rates(
                symbol, market_type, calc_time, funding_interval_hours,
                funding_rate, source, ingested_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, market_type, calc_time) DO UPDATE SET
                funding_interval_hours=excluded.funding_interval_hours,
                funding_rate=excluded.funding_rate,
                source=excluded.source,
                ingested_at_ms=excluded.ingested_at_ms
            WHERE funding_rates.funding_interval_hours IS NOT excluded.funding_interval_hours
               OR funding_rates.funding_rate IS NOT excluded.funding_rate
               OR funding_rates.source IS NOT excluded.source
                """,
                rows,
            )
        return max(0, self.connect().total_changes - before)

    def fetch_futures_reference_bars(
        self,
        symbol: str,
        *,
        kind: str = "premium_index",
        market_type: str = "futures",
        interval: str = "1m",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[FuturesReferenceBar]:
        where = [
            "symbol = ?",
            "market_type = ?",
            "kind = ?",
            "interval = ?",
        ]
        params: list[object] = [symbol.upper(), market_type, kind, interval]
        if start_ms is not None:
            where.append("open_time >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("open_time <= ?")
            params.append(int(end_ms))
        rows = self.connect().execute(
            f"""
            SELECT symbol, market_type, kind, interval, open_time, open, high,
                   low, close, close_time
            FROM futures_reference_bars
            WHERE {" AND ".join(where)}
            ORDER BY open_time
            """,
            params,
        ).fetchall()
        return [
            FuturesReferenceBar(
                symbol=str(row["symbol"]),
                market_type=str(row["market_type"]),
                kind=str(row["kind"]),
                interval=str(row["interval"]),
                open_time=int(row["open_time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                close_time=int(row["close_time"]),
            )
            for row in rows
        ]

    def fetch_funding_rates(
        self,
        symbol: str,
        *,
        market_type: str = "futures",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[FundingRateRecord]:
        where = ["symbol = ?", "market_type = ?"]
        params: list[object] = [symbol.upper(), market_type]
        if start_ms is not None:
            where.append("calc_time >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("calc_time <= ?")
            params.append(int(end_ms))
        rows = self.connect().execute(
            f"""
            SELECT symbol, market_type, calc_time, funding_interval_hours,
                   funding_rate
            FROM funding_rates
            WHERE {" AND ".join(where)}
            ORDER BY calc_time
            """,
            params,
        ).fetchall()
        return [
            FundingRateRecord(
                symbol=str(row["symbol"]),
                market_type=str(row["market_type"]),
                calc_time=int(row["calc_time"]),
                funding_interval_hours=int(row["funding_interval_hours"]),
                funding_rate=float(row["funding_rate"]),
            )
            for row in rows
        ]

    def begin_derivatives_archive_file(
        self,
        *,
        url: str,
        symbol: str,
        market_type: str,
        data_type: str,
        interval: str,
        period: str,
        started_at_ms: int | None = None,
    ) -> None:
        timestamp = self._now_ms() if started_at_ms is None else int(started_at_ms)
        self.connect().execute(
            """
            INSERT INTO derivatives_archive_files(
                url, symbol, market_type, data_type, interval, period, status,
                rows_inserted, rows_read, bytes_downloaded, sha256,
                checksum_sha256, checksum_status, row_stream_sha256, error,
                started_at_ms, completed_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, 'started', 0, 0, 0, '', '',
                    'unverified', '', '', ?, NULL)
            ON CONFLICT(url) DO UPDATE SET
                status='started',
                error='',
                started_at_ms=excluded.started_at_ms,
                completed_at_ms=NULL
            """,
            (
                url,
                symbol.upper(),
                market_type,
                data_type,
                interval,
                period,
                timestamp,
            ),
        )
        self.connect().commit()

    def complete_derivatives_archive_file(
        self,
        *,
        url: str,
        status: str,
        rows_inserted: int,
        rows_read: int,
        bytes_downloaded: int,
        sha256: str,
        checksum_sha256: str,
        checksum_status: str,
        row_stream_sha256: str,
        error: str = "",
        completed_at_ms: int | None = None,
    ) -> None:
        timestamp = self._now_ms() if completed_at_ms is None else int(completed_at_ms)
        self.connect().execute(
            """
            UPDATE derivatives_archive_files
            SET status=?, rows_inserted=?, rows_read=?, bytes_downloaded=?,
                sha256=?, checksum_sha256=?, checksum_status=?,
                row_stream_sha256=?, error=?, completed_at_ms=?
            WHERE url=?
            """,
            (
                status,
                max(0, int(rows_inserted)),
                max(0, int(rows_read)),
                max(0, int(bytes_downloaded)),
                str(sha256 or ""),
                str(checksum_sha256 or ""),
                str(checksum_status or "unverified"),
                str(row_stream_sha256 or ""),
                str(error or "")[:500],
                timestamp,
                url,
            ),
        )
        self.connect().commit()

    def derivatives_archive_file_status(self, url: str) -> str | None:
        row = self.connect().execute(
            "SELECT status FROM derivatives_archive_files WHERE url = ?", (url,)
        ).fetchone()
        return str(row["status"]) if row is not None else None

    def derivatives_archive_files(
        self,
        *,
        symbol: str | None = None,
        data_type: str | None = None,
        status: str | None = None,
    ) -> list[DerivativesArchiveFileRecord]:
        where: list[str] = []
        params: list[object] = []
        if symbol is not None:
            where.append("symbol = ?")
            params.append(symbol.upper())
        if data_type is not None:
            where.append("data_type = ?")
            params.append(data_type)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        query = """
            SELECT url, symbol, market_type, data_type, interval, period, status,
                   rows_inserted, rows_read, bytes_downloaded, sha256,
                   checksum_sha256, checksum_status, row_stream_sha256, error,
                   started_at_ms, completed_at_ms
            FROM derivatives_archive_files
        """
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY symbol, data_type, interval, period"
        rows = self.connect().execute(query, params).fetchall()
        return [
            DerivativesArchiveFileRecord(
                url=str(row["url"]),
                symbol=str(row["symbol"]),
                market_type=str(row["market_type"]),
                data_type=str(row["data_type"]),
                interval=str(row["interval"]),
                period=str(row["period"]),
                status=str(row["status"]),
                rows_inserted=int(row["rows_inserted"]),
                rows_read=int(row["rows_read"]),
                bytes_downloaded=int(row["bytes_downloaded"]),
                sha256=str(row["sha256"]),
                checksum_sha256=str(row["checksum_sha256"]),
                checksum_status=str(row["checksum_status"]),
                row_stream_sha256=str(row["row_stream_sha256"]),
                error=str(row["error"]),
                started_at_ms=int(row["started_at_ms"]),
                completed_at_ms=(
                    int(row["completed_at_ms"])
                    if row["completed_at_ms"] is not None
                    else None
                ),
            )
            for row in rows
        ]
