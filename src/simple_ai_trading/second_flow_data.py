"""Verified one-second Binance taker-flow data for the Round 42 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Mapping

import numpy as np


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MARKET_TYPE = "futures"
INTERVAL = "1s"
START_MS = 1_717_200_000_000
END_EXCLUSIVE_MS = 1_717_804_800_000
EXPECTED_SECONDS = 604_800
EXPECTED_RAW_AGG_TRADES = {
    "BTCUSDT": 6_361_886,
    "ETHUSDT": 6_181_375,
    "SOLUSDT": 2_932_035,
}
SOURCE_SCHEMA = "round-042-second-flow-source-certificate-v1"
STREAM_SCHEMA = "second-flow-canonical-array-stream-v1"


@dataclass(frozen=True)
class SecondFlowSeries:
    symbol: str
    open_time_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    quote_volume: np.ndarray
    trade_count: np.ndarray
    taker_buy_base_volume: np.ndarray
    taker_buy_quote_volume: np.ndarray
    source: tuple[str, ...]

    @property
    def rows(self) -> int:
        return int(self.open_time_ms.size)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _empty_arrays() -> dict[str, np.ndarray]:
    return {
        "open_time_ms": np.empty(EXPECTED_SECONDS, dtype=np.int64),
        "open": np.empty(EXPECTED_SECONDS, dtype=np.float64),
        "high": np.empty(EXPECTED_SECONDS, dtype=np.float64),
        "low": np.empty(EXPECTED_SECONDS, dtype=np.float64),
        "close": np.empty(EXPECTED_SECONDS, dtype=np.float64),
        "volume": np.empty(EXPECTED_SECONDS, dtype=np.float64),
        "quote_volume": np.empty(EXPECTED_SECONDS, dtype=np.float64),
        "trade_count": np.empty(EXPECTED_SECONDS, dtype=np.int64),
        "taker_buy_base_volume": np.empty(EXPECTED_SECONDS, dtype=np.float64),
        "taker_buy_quote_volume": np.empty(EXPECTED_SECONDS, dtype=np.float64),
    }


def _validate_series(series: SecondFlowSeries) -> None:
    expected_times = START_MS + np.arange(EXPECTED_SECONDS, dtype=np.int64) * 1000
    if series.rows != EXPECTED_SECONDS or not np.array_equal(
        series.open_time_ms, expected_times
    ):
        raise ValueError(f"{series.symbol} one-second timestamps are incomplete")
    numeric = (
        series.open,
        series.high,
        series.low,
        series.close,
        series.volume,
        series.quote_volume,
        series.taker_buy_base_volume,
        series.taker_buy_quote_volume,
    )
    if any(not np.isfinite(values).all() for values in numeric):
        raise ValueError(f"{series.symbol} one-second values are nonfinite")
    if (
        np.any(series.open <= 0.0)
        or np.any(series.high < np.maximum(series.open, series.close))
        or np.any(series.low > np.minimum(series.open, series.close))
        or np.any(series.low <= 0.0)
        or np.any(series.volume < 0.0)
        or np.any(series.quote_volume < 0.0)
        or np.any(series.trade_count < 0)
        or np.any(series.taker_buy_base_volume < 0.0)
        or np.any(series.taker_buy_quote_volume < 0.0)
        or np.any(series.taker_buy_base_volume > series.volume + 1e-9)
        or np.any(series.taker_buy_quote_volume > series.quote_volume + 1e-6)
    ):
        raise ValueError(f"{series.symbol} one-second market values are invalid")
    if not series.source or any(not value for value in series.source):
        raise ValueError(f"{series.symbol} one-second source identity is missing")


def _load_symbol(connection: sqlite3.Connection, symbol: str) -> SecondFlowSeries:
    arrays = _empty_arrays()
    sources: set[str] = set()
    cursor = connection.execute(
        """
        SELECT open_time, open, high, low, close, volume, quote_volume,
               trade_count, taker_buy_base_volume, taker_buy_quote_volume, source
        FROM candles
        WHERE symbol = ? AND market_type = ? AND interval = ?
          AND open_time >= ? AND open_time < ?
        ORDER BY open_time
        """,
        (symbol, MARKET_TYPE, INTERVAL, START_MS, END_EXCLUSIVE_MS),
    )
    offset = 0
    while rows := cursor.fetchmany(50_000):
        end = offset + len(rows)
        if end > EXPECTED_SECONDS:
            raise ValueError(f"{symbol} one-second stream exceeds frozen length")
        block = np.asarray([tuple(row[:10]) for row in rows], dtype=np.float64)
        arrays["open_time_ms"][offset:end] = block[:, 0].astype(np.int64)
        arrays["open"][offset:end] = block[:, 1]
        arrays["high"][offset:end] = block[:, 2]
        arrays["low"][offset:end] = block[:, 3]
        arrays["close"][offset:end] = block[:, 4]
        arrays["volume"][offset:end] = block[:, 5]
        arrays["quote_volume"][offset:end] = block[:, 6]
        arrays["trade_count"][offset:end] = block[:, 7].astype(np.int64)
        arrays["taker_buy_base_volume"][offset:end] = block[:, 8]
        arrays["taker_buy_quote_volume"][offset:end] = block[:, 9]
        sources.update(str(row[10]) for row in rows)
        offset = end
    if offset != EXPECTED_SECONDS:
        raise ValueError(
            f"{symbol} has {offset} one-second rows; expected {EXPECTED_SECONDS}"
        )
    series = SecondFlowSeries(
        symbol=symbol,
        source=tuple(sorted(sources)),
        **arrays,
    )
    _validate_series(series)
    return series


def stream_sha256(series: SecondFlowSeries) -> str:
    digest = hashlib.sha256()
    digest.update(STREAM_SCHEMA.encode("ascii"))
    digest.update(series.symbol.encode("ascii"))
    fields = (
        ("open_time_ms", series.open_time_ms, "<i8"),
        ("open", series.open, "<f8"),
        ("high", series.high, "<f8"),
        ("low", series.low, "<f8"),
        ("close", series.close, "<f8"),
        ("volume", series.volume, "<f8"),
        ("quote_volume", series.quote_volume, "<f8"),
        ("trade_count", series.trade_count, "<i8"),
        ("taker_buy_base_volume", series.taker_buy_base_volume, "<f8"),
        ("taker_buy_quote_volume", series.taker_buy_quote_volume, "<f8"),
    )
    for name, values, dtype in fields:
        digest.update(name.encode("ascii"))
        canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
        digest.update(memoryview(canonical).cast("B"))
    digest.update(_canonical_json(series.source))
    return digest.hexdigest()


def _archive_evidence(
    connection: sqlite3.Connection, symbol: str
) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT period, url, status, bytes_downloaded, sha256,
               checksum_sha256, checksum_status
        FROM archive_files
        WHERE symbol = ? AND market_type = ? AND interval = ?
          AND period >= '2024-06-01' AND period <= '2024-06-07'
          AND url LIKE '%/daily/aggTrades/%'
        ORDER BY period, url
        """,
        (symbol, MARKET_TYPE, INTERVAL),
    ).fetchall()
    evidence = [
        {
            "period": str(row["period"]),
            "url": str(row["url"]),
            "status": str(row["status"]),
            "bytes_downloaded": int(row["bytes_downloaded"]),
            "archive_sha256": str(row["sha256"]),
            "expected_checksum_sha256": str(row["checksum_sha256"]),
            "checksum_status": str(row["checksum_status"]),
        }
        for row in rows
    ]
    if (
        len(evidence) != 7
        or [item["period"] for item in evidence]
        != [f"2024-06-{day:02d}" for day in range(1, 8)]
        or any(
            item["status"] != "complete"
            or item["checksum_status"] != "verified"
            or len(str(item["archive_sha256"])) != 64
            or item["archive_sha256"] != item["expected_checksum_sha256"]
            for item in evidence
        )
    ):
        raise ValueError(f"{symbol} frozen archive evidence is incomplete")
    return evidence


def _raw_trade_evidence(
    connection: sqlite3.Connection, symbol: str
) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT count(*) AS rows, min(trade_time_ms) AS first_ms,
               max(trade_time_ms) AS last_ms,
               count(DISTINCT CAST(trade_time_ms / 1000 AS INTEGER)) AS active_seconds,
               count(DISTINCT source) AS source_count
        FROM agg_trades
        WHERE symbol = ? AND market_type = ?
          AND trade_time_ms >= ? AND trade_time_ms < ?
        """,
        (symbol, MARKET_TYPE, START_MS, END_EXCLUSIVE_MS),
    ).fetchone()
    evidence = {
        "rows": int(row["rows"]),
        "first_trade_time_ms": int(row["first_ms"]),
        "last_trade_time_ms": int(row["last_ms"]),
        "active_seconds": int(row["active_seconds"]),
        "source_count": int(row["source_count"]),
    }
    if (
        evidence["rows"] != EXPECTED_RAW_AGG_TRADES[symbol]
        or not START_MS <= evidence["first_trade_time_ms"] < END_EXCLUSIVE_MS
        or not START_MS <= evidence["last_trade_time_ms"] < END_EXCLUSIVE_MS
        or evidence["source_count"] != 1
    ):
        raise ValueError(f"{symbol} raw aggregate-trade coverage drifted")
    return evidence


def load_verified_second_flow(
    database: Path,
) -> tuple[dict[str, SecondFlowSeries], dict[str, object]]:
    series_by_symbol: dict[str, SecondFlowSeries] = {}
    symbol_evidence: list[dict[str, object]] = []
    with _connect_read_only(database) as connection:
        for symbol in SYMBOLS:
            series = _load_symbol(connection, symbol)
            archives = _archive_evidence(connection, symbol)
            raw = _raw_trade_evidence(connection, symbol)
            series_by_symbol[symbol] = series
            symbol_evidence.append(
                {
                    "symbol": symbol,
                    "rows": series.rows,
                    "first_open_time_ms": int(series.open_time_ms[0]),
                    "last_open_time_ms": int(series.open_time_ms[-1]),
                    "gap_count": int(
                        np.count_nonzero(np.diff(series.open_time_ms) != 1000)
                    ),
                    "zero_trade_seconds": int(
                        np.count_nonzero(series.trade_count == 0)
                    ),
                    "source": list(series.source),
                    "stream_schema": STREAM_SCHEMA,
                    "stream_sha256": stream_sha256(series),
                    "raw_aggregate_trades": raw,
                    "archives": archives,
                    "archive_manifest_sha256": canonical_sha256(archives),
                }
            )
    evidence: dict[str, object] = {
        "schema_version": SOURCE_SCHEMA,
        "market": "Binance USD-M perpetual futures",
        "market_type": MARKET_TYPE,
        "interval": INTERVAL,
        "period_start_ms": START_MS,
        "period_end_exclusive_ms": END_EXCLUSIVE_MS,
        "period_start_utc": datetime.fromtimestamp(START_MS / 1000, UTC).isoformat(),
        "period_end_exclusive_utc": datetime.fromtimestamp(
            END_EXCLUSIVE_MS / 1000, UTC
        ).isoformat(),
        "database_path": str(database.resolve()),
        "database_opened_read_only": True,
        "symbols": symbol_evidence,
        "rows_total": sum(item["rows"] for item in symbol_evidence),
        "raw_aggregate_trade_rows_total": sum(
            item["raw_aggregate_trades"]["rows"] for item in symbol_evidence
        ),
        "certificate_sha256": "PENDING",
    }
    canonical = dict(evidence)
    canonical.pop("certificate_sha256")
    evidence["certificate_sha256"] = canonical_sha256(canonical)
    return series_by_symbol, evidence


def validate_source_certificate(
    certificate: Mapping[str, object], current: Mapping[str, object]
) -> None:
    claimed = str(certificate.get("certificate_sha256") or "")
    canonical = dict(certificate)
    canonical.pop("certificate_sha256", None)
    if (
        certificate.get("schema_version") != SOURCE_SCHEMA
        or len(claimed) != 64
        or canonical_sha256(canonical) != claimed
        or dict(certificate) != dict(current)
    ):
        raise ValueError("Round 42 one-second source certificate drifted")


__all__ = [
    "END_EXCLUSIVE_MS",
    "EXPECTED_SECONDS",
    "INTERVAL",
    "MARKET_TYPE",
    "SOURCE_SCHEMA",
    "START_MS",
    "SYMBOLS",
    "SecondFlowSeries",
    "canonical_sha256",
    "file_sha256",
    "load_verified_second_flow",
    "stream_sha256",
    "validate_source_certificate",
]
