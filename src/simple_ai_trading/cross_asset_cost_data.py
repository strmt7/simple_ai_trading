"""Causal multi-year BTC/ETH/SOL minute dataset for cost-aware research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
import sqlite3
from typing import Callable, Mapping, Sequence

import numpy as np


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HORIZONS_MINUTES = (15, 30, 60, 120)
DECISION_CADENCE_MINUTES = 5
MINUTE_MS = 60_000
FEATURE_WARMUP_START = "2021-12-01"
MATERIALIZATION_END = "2025-06-30"


@dataclass(frozen=True)
class ChronologicalRole:
    name: str
    start: str
    end: str
    targets_permitted: bool

    @property
    def start_ms(self) -> int:
        return _date_ms(self.start)

    @property
    def end_exclusive_ms(self) -> int:
        return _date_ms(self.end) + 86_400_000


ROLES = (
    ChronologicalRole("training", "2022-01-01", "2024-06-30", True),
    ChronologicalRole("early_stop", "2024-07-01", "2024-09-30", True),
    ChronologicalRole("calibration", "2024-10-01", "2024-12-31", True),
    ChronologicalRole("viability", "2025-01-01", "2025-06-30", True),
    ChronologicalRole(
        "selection_confirmation",
        "2025-07-01",
        "2025-12-31",
        False,
    ),
    ChronologicalRole("terminal", "2026-01-01", "2026-06-30", False),
)


@dataclass(frozen=True)
class ArchiveEvidence:
    symbol: str
    complete_verified_archives: int
    first_period: str
    last_period: str
    archived_rows: int
    non_verified_or_incomplete_archives: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SeriesEvidence:
    symbol: str
    rows: int
    first_open_time_ms: int
    last_open_time_ms: int
    gap_count: int
    duplicate_or_regressed_time_count: int
    nonfinite_numeric_rows: int
    invalid_ohlc_rows: int
    invalid_volume_rows: int
    source_counts: Mapping[str, int]
    stream_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceEvidence:
    database_path: str
    materialized_start: str
    materialized_end: str
    archive_evidence: tuple[ArchiveEvidence, ...]
    series_evidence: tuple[SeriesEvidence, ...]
    panel_stream_sha256: str
    selection_confirmation_or_terminal_rows_read: bool

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "archive_evidence": [item.asdict() for item in self.archive_evidence],
            "series_evidence": [item.asdict() for item in self.series_evidence],
        }


@dataclass(frozen=True)
class MinuteSeries:
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


@dataclass(frozen=True)
class CrossAssetDataset:
    feature_names: tuple[str, ...]
    features: np.ndarray
    decision_time_ms: np.ndarray
    symbol_index: np.ndarray
    gross_return_bps: Mapping[int, np.ndarray]
    persistence_prediction_bps: Mapping[int, np.ndarray]
    role_masks: Mapping[int, Mapping[str, np.ndarray]]
    source_evidence: SourceEvidence

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _date_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def _read_only_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"market database not found: {resolved}")
    uri = f"file:{resolved.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def _archive_evidence(
    connection: sqlite3.Connection,
    symbol: str,
) -> ArchiveEvidence:
    row = connection.execute(
        """
        SELECT
            COUNT(*),
            MIN(period),
            MAX(period),
            COALESCE(SUM(rows_inserted), 0)
        FROM archive_files
        WHERE symbol = ?
          AND market_type = 'futures'
          AND interval = '1m'
          AND status = 'complete'
          AND checksum_status = 'verified'
        """,
        (symbol,),
    ).fetchone()
    invalid = connection.execute(
        """
        SELECT COUNT(*)
        FROM archive_files
        WHERE symbol = ?
          AND market_type = 'futures'
          AND interval = '1m'
          AND NOT (status = 'complete' AND checksum_status = 'verified')
        """,
        (symbol,),
    ).fetchone()
    if row is None or int(row[0]) <= 0 or not row[1] or not row[2]:
        raise ValueError(f"{symbol} has no complete checksum-verified 1m archives")
    return ArchiveEvidence(
        symbol=symbol,
        complete_verified_archives=int(row[0]),
        first_period=str(row[1]),
        last_period=str(row[2]),
        archived_rows=int(row[3]),
        non_verified_or_incomplete_archives=int(invalid[0]) if invalid else 0,
    )


def _source_hash_update(
    digest: "hashlib._Hash",
    *,
    symbol: str,
    numeric: np.ndarray,
    sources: Sequence[str],
) -> None:
    digest.update(symbol.encode("ascii"))
    digest.update(b"\x00")
    digest.update(int(numeric.shape[0]).to_bytes(8, "little", signed=False))
    digest.update(np.ascontiguousarray(numeric).tobytes(order="C"))
    for source in sources:
        encoded = source.encode("utf-8")
        digest.update(len(encoded).to_bytes(2, "little", signed=False))
        digest.update(encoded)


def _load_series(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    start_ms: int,
    end_exclusive_ms: int,
    progress: ProgressCallback | None,
) -> tuple[MinuteSeries, SeriesEvidence]:
    count_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM candles
        WHERE symbol = ? AND market_type = 'futures' AND interval = '1m'
          AND open_time >= ? AND open_time < ?
        """,
        (symbol, start_ms, end_exclusive_ms),
    ).fetchone()
    rows = int(count_row[0]) if count_row else 0
    expected = (end_exclusive_ms - start_ms) // MINUTE_MS
    if rows != expected:
        raise ValueError(
            f"{symbol} minute coverage mismatch: rows={rows} expected={expected}"
        )

    open_time_ms = np.empty(rows, dtype=np.int64)
    open_values = np.empty(rows, dtype=np.float64)
    high_values = np.empty(rows, dtype=np.float64)
    low_values = np.empty(rows, dtype=np.float64)
    close_values = np.empty(rows, dtype=np.float64)
    volume = np.empty(rows, dtype=np.float64)
    quote_volume = np.empty(rows, dtype=np.float64)
    trade_count = np.empty(rows, dtype=np.int64)
    taker_buy_base = np.empty(rows, dtype=np.float64)
    taker_buy_quote = np.empty(rows, dtype=np.float64)
    source_counts: dict[str, int] = {}
    digest = hashlib.sha256()

    cursor = connection.execute(
        """
        SELECT
            open_time, open, high, low, close, volume, quote_volume,
            trade_count, taker_buy_base_volume, taker_buy_quote_volume,
            source
        FROM candles
        WHERE symbol = ? AND market_type = 'futures' AND interval = '1m'
          AND open_time >= ? AND open_time < ?
        ORDER BY open_time
        """,
        (symbol, start_ms, end_exclusive_ms),
    )
    offset = 0
    batch_size = 100_000
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        count = len(batch)
        numeric = np.asarray([row[:-1] for row in batch], dtype=np.float64)
        timestamps = numeric[:, 0].astype(np.int64)
        trades = numeric[:, 7].astype(np.int64)
        sources = [str(row[-1]) for row in batch]
        target = slice(offset, offset + count)
        open_time_ms[target] = timestamps
        open_values[target] = numeric[:, 1]
        high_values[target] = numeric[:, 2]
        low_values[target] = numeric[:, 3]
        close_values[target] = numeric[:, 4]
        volume[target] = numeric[:, 5]
        quote_volume[target] = numeric[:, 6]
        trade_count[target] = trades
        taker_buy_base[target] = numeric[:, 8]
        taker_buy_quote[target] = numeric[:, 9]
        hash_numeric = np.column_stack((timestamps, numeric[:, 1:7], trades, numeric[:, 8:10]))
        _source_hash_update(
            digest,
            symbol=symbol,
            numeric=hash_numeric,
            sources=sources,
        )
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        offset += count
        if progress is not None:
            progress(
                "source_load",
                {"symbol": symbol, "rows_loaded": offset, "rows_total": rows},
            )
    if offset != rows:
        raise RuntimeError(f"{symbol} source load ended at {offset} of {rows} rows")

    time_deltas = np.diff(open_time_ms)
    gap_count = int(np.count_nonzero(time_deltas > MINUTE_MS))
    duplicate_or_regressed = int(np.count_nonzero(time_deltas <= 0))
    numeric_columns = (
        open_values,
        high_values,
        low_values,
        close_values,
        volume,
        quote_volume,
        taker_buy_base,
        taker_buy_quote,
    )
    nonfinite = int(
        np.count_nonzero(~np.logical_and.reduce([np.isfinite(value) for value in numeric_columns]))
    )
    invalid_ohlc = int(
        np.count_nonzero(
            (open_values <= 0.0)
            | (high_values <= 0.0)
            | (low_values <= 0.0)
            | (close_values <= 0.0)
            | (high_values < np.maximum(open_values, close_values))
            | (low_values > np.minimum(open_values, close_values))
            | (high_values < low_values)
        )
    )
    invalid_volume = int(
        np.count_nonzero(
            (volume < 0.0)
            | (quote_volume < 0.0)
            | (trade_count < 0)
            | (taker_buy_base < 0.0)
            | (taker_buy_quote < 0.0)
            | (taker_buy_base > volume + 1e-9)
            | (taker_buy_quote > quote_volume + 1e-6)
        )
    )
    if gap_count or duplicate_or_regressed or nonfinite or invalid_ohlc or invalid_volume:
        raise ValueError(
            f"{symbol} source failed integrity checks: gaps={gap_count}, "
            f"duplicates={duplicate_or_regressed}, nonfinite={nonfinite}, "
            f"ohlc={invalid_ohlc}, volume={invalid_volume}"
        )
    evidence = SeriesEvidence(
        symbol=symbol,
        rows=rows,
        first_open_time_ms=int(open_time_ms[0]),
        last_open_time_ms=int(open_time_ms[-1]),
        gap_count=gap_count,
        duplicate_or_regressed_time_count=duplicate_or_regressed,
        nonfinite_numeric_rows=nonfinite,
        invalid_ohlc_rows=invalid_ohlc,
        invalid_volume_rows=invalid_volume,
        source_counts=dict(sorted(source_counts.items())),
        stream_sha256=digest.hexdigest(),
    )
    return (
        MinuteSeries(
            symbol=symbol,
            open_time_ms=open_time_ms,
            open=open_values,
            high=high_values,
            low=low_values,
            close=close_values,
            volume=volume,
            quote_volume=quote_volume,
            trade_count=trade_count,
            taker_buy_base_volume=taker_buy_base,
            taker_buy_quote_volume=taker_buy_quote,
        ),
        evidence,
    )


def load_verified_minute_panel(
    database_path: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, MinuteSeries], SourceEvidence]:
    """Load only the frozen warmup-through-viability source window."""

    path = Path(database_path)
    start_ms = _date_ms(FEATURE_WARMUP_START)
    end_exclusive_ms = _date_ms(MATERIALIZATION_END) + 86_400_000
    with _read_only_connection(path) as connection:
        archives = tuple(_archive_evidence(connection, symbol) for symbol in SYMBOLS)
        panel: dict[str, MinuteSeries] = {}
        evidence: list[SeriesEvidence] = []
        for symbol in SYMBOLS:
            series, series_evidence = _load_series(
                connection,
                symbol=symbol,
                start_ms=start_ms,
                end_exclusive_ms=end_exclusive_ms,
                progress=progress,
            )
            panel[symbol] = series
            evidence.append(series_evidence)

    timestamps = panel[SYMBOLS[0]].open_time_ms
    for symbol in SYMBOLS[1:]:
        if not np.array_equal(timestamps, panel[symbol].open_time_ms):
            raise ValueError(f"{symbol} minute timestamps do not align with BTCUSDT")
    panel_digest = hashlib.sha256()
    for item in evidence:
        panel_digest.update(item.symbol.encode("ascii"))
        panel_digest.update(bytes.fromhex(item.stream_sha256))
    source = SourceEvidence(
        database_path=str(path.resolve()),
        materialized_start=FEATURE_WARMUP_START,
        materialized_end=MATERIALIZATION_END,
        archive_evidence=archives,
        series_evidence=tuple(evidence),
        panel_stream_sha256=panel_digest.hexdigest(),
        selection_confirmation_or_terminal_rows_read=False,
    )
    return panel, source


def _lagged_return_bps(log_price: np.ndarray, window: int) -> np.ndarray:
    result = np.full(log_price.size, np.nan, dtype=np.float64)
    result[window:] = (log_price[window:] - log_price[:-window]) * 10_000.0
    return result


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.size, np.nan, dtype=np.float64)
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    prefix = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    result[window - 1 :] = prefix[window:] - prefix[:-window]
    return result


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_sum(values, window) / float(window)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.size, np.nan, dtype=np.float64)
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    prefix = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    squared = np.concatenate(([0.0], np.cumsum(clean * clean, dtype=np.float64)))
    total = prefix[window:] - prefix[:-window]
    total_sq = squared[window:] - squared[:-window]
    variance = np.maximum(total_sq / window - (total / window) ** 2, 0.0)
    result[window - 1 :] = np.sqrt(variance)
    return result


def _rolling_covariance(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    left_mean = _rolling_mean(left, window)
    right_mean = _rolling_mean(right, window)
    cross_mean = _rolling_mean(left * right, window)
    return cross_mean - left_mean * right_mean


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.isfinite(denominator) & (np.abs(denominator) > 1e-12),
    )
    return result


def _rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    mean = _rolling_mean(values, window)
    standard_deviation = _rolling_std(values, window)
    return _safe_ratio(values - mean, standard_deviation)


def _path_efficiency(log_price: np.ndarray, returns_1m_bps: np.ndarray, window: int) -> np.ndarray:
    displacement = np.abs(_lagged_return_bps(log_price, window))
    path = _rolling_sum(np.abs(returns_1m_bps), window)
    return _safe_ratio(displacement, path)


def _past_week_liquidity_ratio(values: np.ndarray) -> np.ndarray:
    week = 7 * 24 * 60
    history = np.full((4, values.size), np.nan, dtype=np.float64)
    for index in range(4):
        lag = week * (index + 1)
        history[index, lag:] = values[:-lag]
    counts = np.sum(np.isfinite(history), axis=0)
    historical_mean = np.full(values.size, np.nan, dtype=np.float64)
    np.divide(
        np.nansum(history, axis=0),
        counts,
        out=historical_mean,
        where=counts > 0,
    )
    return _safe_ratio(values, historical_mean)


def _cross_sectional_mean(values: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(
        np.nansum(values, axis=0),
        counts,
        out=result,
        where=counts > 0,
    )
    return result


def _cross_sectional_std(values: np.ndarray) -> np.ndarray:
    mean = _cross_sectional_mean(values)
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=0)
    centered = np.where(valid, values - mean, 0.0)
    variance = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(
        np.sum(centered * centered, axis=0),
        counts,
        out=variance,
        where=counts > 0,
    )
    return np.sqrt(variance)


def _feature_arrays(
    panel: Mapping[str, MinuteSeries],
    target_symbol: str,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    target = panel[target_symbol]
    log_close = {symbol: np.log(panel[symbol].close) for symbol in SYMBOLS}
    return_1m = {
        symbol: _lagged_return_bps(log_close[symbol], 1) for symbol in SYMBOLS
    }
    target_returns = return_1m[target_symbol]
    names: list[str] = []
    values: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        values.append(value)

    for window in (1, 3, 5, 15, 30, 60, 240, 1440):
        add(f"target_return_{window}m_bps", _lagged_return_bps(log_close[target_symbol], window))
    for window in (5, 15, 60, 240, 1440):
        add(f"target_realized_volatility_{window}m_bps", _rolling_std(target_returns, window))
    negative = np.minimum(target_returns, 0.0)
    positive = np.maximum(target_returns, 0.0)
    for window in (60, 240):
        add(f"target_downside_semivolatility_{window}m_bps", np.sqrt(_rolling_mean(negative * negative, window)))
        add(f"target_upside_semivolatility_{window}m_bps", np.sqrt(_rolling_mean(positive * positive, window)))

    range_bps = np.log(target.high / target.low) * 10_000.0
    close_location = _safe_ratio(target.close - target.low, target.high - target.low) - 0.5
    add("target_intrabar_range_bps", range_bps)
    add("target_close_location", close_location)
    for window in (15, 60, 240):
        add(f"target_path_efficiency_{window}m", _path_efficiency(log_close[target_symbol], target_returns, window))

    log_volume = np.log1p(target.volume)
    log_quote_volume = np.log1p(target.quote_volume)
    log_trade_count = np.log1p(target.trade_count.astype(np.float64))
    add("target_log_base_volume", log_volume)
    add("target_log_quote_volume", log_quote_volume)
    add("target_log_trade_count", log_trade_count)
    for label, series in (("quote_volume", target.quote_volume), ("trade_count", target.trade_count.astype(np.float64))):
        for window in (60, 240, 1440):
            add(f"target_{label}_vs_{window}m_mean", _safe_ratio(series, _rolling_mean(series, window)))

    taker_share = 2.0 * _safe_ratio(target.taker_buy_quote_volume, target.quote_volume) - 1.0
    signed_taker_quote = target.quote_volume * taker_share
    add("target_taker_buy_minus_sell_share", taker_share)
    for window in (1, 5, 15, 60):
        signed = signed_taker_quote if window == 1 else _rolling_sum(signed_taker_quote, window)
        total = target.quote_volume if window == 1 else _rolling_sum(target.quote_volume, window)
        add(f"target_signed_taker_flow_{window}m", _safe_ratio(signed, total))
    add("target_return_zscore_240m", _rolling_zscore(target_returns, 240))
    add("target_quote_volume_zscore_240m", _rolling_zscore(log_quote_volume, 240))
    add("target_range_zscore_240m", _rolling_zscore(range_bps, 240))
    add("target_same_minute_of_week_liquidity_ratio", _past_week_liquidity_ratio(target.quote_volume))

    for symbol in SYMBOLS:
        for window in (1, 5, 15, 60):
            add(f"{symbol.lower()}_return_{window}m_bps", _lagged_return_bps(log_close[symbol], window))
    if target_symbol == "BTCUSDT":
        benchmark_return = (return_1m["ETHUSDT"] + return_1m["SOLUSDT"]) / 2.0
        benchmark_log = (log_close["ETHUSDT"] + log_close["SOLUSDT"]) / 2.0
    else:
        benchmark_return = return_1m["BTCUSDT"]
        benchmark_log = log_close["BTCUSDT"]
    for window in (60, 240, 1440):
        covariance = _rolling_covariance(target_returns, benchmark_return, window)
        variance = _rolling_covariance(benchmark_return, benchmark_return, window)
        beta = _safe_ratio(covariance, variance)
        target_window_return = _lagged_return_bps(log_close[target_symbol], window)
        benchmark_window_return = _lagged_return_bps(benchmark_log, window)
        add(f"target_beta_residual_return_{window}m_bps", target_window_return - beta * benchmark_window_return)

    stacked_returns = np.vstack([return_1m[symbol] for symbol in SYMBOLS])
    cross_mean = _cross_sectional_mean(stacked_returns)
    for window in (1, 15, 60):
        if window == 1:
            dispersion = _cross_sectional_std(stacked_returns)
        else:
            rolling_returns = np.vstack(
                [_lagged_return_bps(log_close[symbol], window) for symbol in SYMBOLS]
            )
            dispersion = _cross_sectional_std(rolling_returns)
        add(f"cross_asset_return_dispersion_{window}m_bps", dispersion)
    flow_matrix = np.vstack(
        [
            2.0 * _safe_ratio(panel[symbol].taker_buy_quote_volume, panel[symbol].quote_volume) - 1.0
            for symbol in SYMBOLS
        ]
    )
    add("cross_asset_taker_flow_mean", np.nanmean(flow_matrix, axis=0))
    add("cross_asset_taker_flow_agreement", np.abs(np.nanmean(np.sign(flow_matrix), axis=0)))
    add("target_return_vs_cross_mean_1m_bps", target_returns - cross_mean)

    btc_volatility = _rolling_std(return_1m["BTCUSDT"], 60)
    target_volatility = _rolling_std(target_returns, 60)
    add("target_to_btc_volatility_ratio_60m", _safe_ratio(target_volatility, btc_volatility))
    btc_liquidity = _rolling_mean(panel["BTCUSDT"].quote_volume, 60)
    target_liquidity = _rolling_mean(target.quote_volume, 60)
    add("target_to_btc_liquidity_ratio_60m", _safe_ratio(target_liquidity, btc_liquidity))

    minute = (target.open_time_ms // MINUTE_MS).astype(np.int64)
    minute_of_day = minute % 1440
    day_of_week = (minute // 1440 + 3) % 7
    add("utc_minute_of_day_sin", np.sin(2.0 * math.pi * minute_of_day / 1440.0))
    add("utc_minute_of_day_cos", np.cos(2.0 * math.pi * minute_of_day / 1440.0))
    add("utc_day_of_week_sin", np.sin(2.0 * math.pi * day_of_week / 7.0))
    add("utc_day_of_week_cos", np.cos(2.0 * math.pi * day_of_week / 7.0))
    add("weekend_flag", (day_of_week >= 5).astype(np.float64))
    for symbol in SYMBOLS:
        add(f"symbol_{symbol.lower()}", np.full(target.close.size, float(symbol == target_symbol)))
    return tuple(names), tuple(values)


def _role_masks(
    decision_time_ms: np.ndarray,
    horizon_minutes: int,
) -> dict[str, np.ndarray]:
    exit_time_ms = decision_time_ms + (horizon_minutes + 1) * MINUTE_MS
    masks: dict[str, np.ndarray] = {}
    for role in ROLES:
        if not role.targets_permitted:
            masks[role.name] = np.zeros(decision_time_ms.size, dtype=bool)
            continue
        masks[role.name] = (
            (decision_time_ms >= role.start_ms)
            & (decision_time_ms < role.end_exclusive_ms)
            & (exit_time_ms < role.end_exclusive_ms)
        )
    return masks


def build_cross_asset_dataset(
    panel: Mapping[str, MinuteSeries],
    source_evidence: SourceEvidence,
    *,
    progress: ProgressCallback | None = None,
) -> CrossAssetDataset:
    """Build a memory-only causal decision matrix and gross-return labels."""

    reference = panel[SYMBOLS[0]]
    decision_indices = np.flatnonzero(
        (reference.open_time_ms >= ROLES[0].start_ms)
        & (reference.open_time_ms < ROLES[3].end_exclusive_ms)
        & ((reference.open_time_ms // MINUTE_MS) % DECISION_CADENCE_MINUTES == 0)
    )
    feature_blocks: list[np.ndarray] = []
    time_blocks: list[np.ndarray] = []
    symbol_blocks: list[np.ndarray] = []
    gross_blocks: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    persistence_blocks: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    expected_names: tuple[str, ...] | None = None

    for symbol_index, symbol in enumerate(SYMBOLS):
        if progress is not None:
            progress("feature_build", {"symbol": symbol, "status": "started"})
        names, arrays = _feature_arrays(panel, symbol)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError(f"feature order differs for {symbol}")
        block = np.column_stack([value[decision_indices] for value in arrays]).astype(
            np.float32,
            copy=False,
        )
        if not np.isfinite(block).all():
            invalid = int(np.count_nonzero(~np.isfinite(block)))
            raise ValueError(f"{symbol} feature matrix contains {invalid} non-finite values")
        feature_blocks.append(block)
        decision_times = panel[symbol].open_time_ms[decision_indices].copy()
        time_blocks.append(decision_times)
        symbol_blocks.append(
            np.full(decision_indices.size, symbol_index, dtype=np.int8)
        )
        log_close = np.log(panel[symbol].close)
        for horizon in HORIZONS_MINUTES:
            entry_indices = decision_indices + 1
            exit_indices = entry_indices + horizon
            valid = exit_indices < panel[symbol].open.size
            gross = np.full(decision_indices.size, np.nan, dtype=np.float64)
            gross[valid] = (
                np.log(panel[symbol].open[exit_indices[valid]])
                - np.log(panel[symbol].open[entry_indices[valid]])
            ) * 10_000.0
            persistence = (
                log_close[decision_indices]
                - log_close[decision_indices - horizon]
            ) * 10_000.0
            gross_blocks[horizon].append(gross.astype(np.float32))
            persistence_blocks[horizon].append(persistence.astype(np.float32))
        if progress is not None:
            progress(
                "feature_build",
                {
                    "symbol": symbol,
                    "status": "complete",
                    "decision_rows": int(block.shape[0]),
                    "feature_count": int(block.shape[1]),
                },
            )

    features = np.concatenate(feature_blocks, axis=0)
    decision_time_ms = np.concatenate(time_blocks)
    symbol_index = np.concatenate(symbol_blocks)
    gross_return_bps = {
        horizon: np.concatenate(blocks) for horizon, blocks in gross_blocks.items()
    }
    persistence = {
        horizon: np.concatenate(blocks)
        for horizon, blocks in persistence_blocks.items()
    }
    role_masks = {
        horizon: _role_masks(decision_time_ms, horizon)
        for horizon in HORIZONS_MINUTES
    }
    if expected_names is None:
        raise RuntimeError("no feature blocks were built")
    return CrossAssetDataset(
        feature_names=expected_names,
        features=features,
        decision_time_ms=decision_time_ms,
        symbol_index=symbol_index,
        gross_return_bps=gross_return_bps,
        persistence_prediction_bps=persistence,
        role_masks=role_masks,
        source_evidence=source_evidence,
    )


def utc_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC).date().isoformat()


def utc_month(timestamp_ms: int) -> str:
    value = datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
    return f"{value.year:04d}-{value.month:02d}"


def role_by_name(name: str) -> ChronologicalRole:
    for role in ROLES:
        if role.name == name:
            return role
    raise KeyError(name)


__all__ = [
    "ArchiveEvidence",
    "ChronologicalRole",
    "CrossAssetDataset",
    "DECISION_CADENCE_MINUTES",
    "FEATURE_WARMUP_START",
    "HORIZONS_MINUTES",
    "MATERIALIZATION_END",
    "MINUTE_MS",
    "MinuteSeries",
    "ROLES",
    "SYMBOLS",
    "SeriesEvidence",
    "SourceEvidence",
    "build_cross_asset_dataset",
    "load_verified_minute_panel",
    "role_by_name",
    "utc_date",
    "utc_month",
]
