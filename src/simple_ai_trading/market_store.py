"""SQLite market-data store for candles and auxiliary exchange metrics."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
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

            CREATE TABLE IF NOT EXISTS sync_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.commit()

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
        limit: int | None = None,
    ) -> list[Candle]:
        params: list[object] = [symbol.upper(), market_type, interval]
        query = """
            SELECT open_time, open, high, low, close, volume, close_time,
                   quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume
            FROM candles
            WHERE symbol = ? AND market_type = ? AND interval = ?
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

    def coverage(self, symbol: str, market_type: str, interval: str) -> CandleCoverage:
        row = self.connect().execute(
            """
            SELECT COUNT(*) AS count, MIN(open_time) AS first_open_time, MAX(open_time) AS last_open_time
            FROM candles
            WHERE symbol = ? AND market_type = ? AND interval = ?
            """,
            (symbol.upper(), market_type, interval),
        ).fetchone()
        return CandleCoverage(
            symbol=symbol.upper(),
            market_type=market_type,
            interval=interval,
            count=int(row["count"]),
            first_open_time=row["first_open_time"],
            last_open_time=row["last_open_time"],
        )

    def coverage_quality(
        self,
        symbol: str,
        market_type: str,
        interval: str,
        interval_ms: int,
    ) -> CandleCoverageQuality:
        coverage = self.coverage(symbol, market_type, interval)
        if coverage.count == 0:
            return CandleCoverageQuality(coverage, expected_count=0, gap_count=0, coverage_ratio=0.0)
        if interval_ms <= 0:
            return CandleCoverageQuality(
                coverage,
                expected_count=coverage.count,
                gap_count=0,
                coverage_ratio=1.0,
            )

        rows = self.connect().execute(
            """
            SELECT open_time
            FROM candles
            WHERE symbol = ? AND market_type = ? AND interval = ?
            ORDER BY open_time ASC
            """,
            (symbol.upper(), market_type, interval),
        ).fetchall()
        open_times = [int(row["open_time"]) for row in rows if row["open_time"] is not None]
        missing = 0
        for previous, current in pairwise(open_times):
            delta = current - previous
            if delta > interval_ms:
                missing += max(0, (delta // interval_ms) - 1)

        first_open_time = cast(int, coverage.first_open_time)
        last_open_time = cast(int, coverage.last_open_time)
        span = last_open_time - first_open_time
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
