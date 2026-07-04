"""Data-span and resolution evidence for financial backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

from .api import Candle
from .intervals import interval_milliseconds

_MS_PER_DAY = 24 * 60 * 60 * 1000
_MS_PER_YEAR = int(365.25 * _MS_PER_DAY)


@dataclass(frozen=True)
class DataCoverageReport:
    symbol: str
    market_type: str
    interval: str
    source_scope: str
    expected_interval_ms: int
    integrity_status: str
    integrity_warnings: tuple[str, ...]
    truth_basis: tuple[str, ...]
    full_history_requested: bool
    full_available_history_used: bool
    candles_available: int
    candles_used: int
    rows_used: int
    requested_start_ms: int | None
    requested_end_ms: int | None
    available_start_ms: int | None
    available_end_ms: int | None
    used_start_ms: int | None
    used_end_ms: int | None
    available_start_utc: str | None
    available_end_utc: str | None
    used_start_utc: str | None
    used_end_utc: str | None
    used_duration_days: float
    used_duration_years: float
    gap_count: int
    largest_gap_ms: int
    largest_gap_intervals: float
    coverage_ratio: float
    notes: tuple[str, ...] = ()

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def iso_utc(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _span(candles: Sequence[Candle]) -> tuple[int | None, int | None]:
    if not candles:
        return None, None
    ordered = sorted(candles, key=lambda candle: int(candle.close_time))
    return int(ordered[0].close_time), int(ordered[-1].close_time)


def _gap_stats(candles: Sequence[Candle], expected_ms: int) -> tuple[int, int, float, float]:
    if len(candles) < 2 or expected_ms <= 0:
        return 0, 0, 0.0, 1.0 if candles else 0.0
    ordered = sorted(candles, key=lambda candle: int(candle.close_time))
    gaps = [
        max(0, int(right.close_time) - int(left.close_time))
        for left, right in zip(ordered, ordered[1:])
    ]
    largest_gap = max(gaps, default=0)
    gap_count = sum(1 for gap in gaps if gap > int(expected_ms * 1.5))
    start = int(ordered[0].close_time)
    end = int(ordered[-1].close_time)
    expected_points = int((end - start) // expected_ms) + 1 if end >= start else len(ordered)
    coverage_ratio = min(1.0, len(ordered) / max(1, expected_points))
    return gap_count, largest_gap, largest_gap / expected_ms, coverage_ratio


def describe_candle_coverage(
    *,
    symbol: str,
    market_type: str,
    interval: str,
    available_candles: Sequence[Candle],
    used_candles: Sequence[Candle],
    rows_used: int,
    requested_start_ms: int | None = None,
    requested_end_ms: int | None = None,
    source_scope: str = "loaded_candles",
) -> DataCoverageReport:
    expected_ms = interval_milliseconds(interval)
    scope = str(source_scope or "loaded_candles")
    recent_limited_scope = scope in {"binance_recent_limit", "api_recent_limit", "recent_limit"}
    available_start, available_end = _span(available_candles)
    used_start, used_end = _span(used_candles)
    full_requested = requested_start_ms is None and requested_end_ms is None
    full_used = bool(
        full_requested
        and not recent_limited_scope
        and available_start == used_start
        and available_end == used_end
        and len(available_candles) == len(used_candles)
    )
    duration_ms = max(0, int((used_end or 0) - (used_start or 0))) if used_start is not None and used_end is not None else 0
    gap_count, largest_gap_ms, largest_gap_intervals, coverage_ratio = _gap_stats(used_candles, expected_ms)
    notes: list[str] = []
    if not used_candles:
        notes.append("no_candles_used")
    if rows_used <= 0:
        notes.append("no_model_rows_used")
    if full_requested and not full_used:
        notes.append("full_requested_but_filtered_data_did_not_cover_all_available_candles")
    if not full_requested:
        notes.append("operator_requested_bounded_window")
    if gap_count:
        notes.append("coverage_gaps_detected")
    if coverage_ratio < 0.995 and used_candles:
        notes.append("coverage_ratio_below_99_5_percent")
    if duration_ms and duration_ms < _MS_PER_YEAR:
        notes.append("less_than_one_year_of_used_history")
    if recent_limited_scope:
        notes.append("recent_api_limit_not_full_history")
    hard_failures = {
        "no_candles_used",
        "no_model_rows_used",
        "coverage_gaps_detected",
        "coverage_ratio_below_99_5_percent",
    }
    integrity_warnings = tuple(dict.fromkeys(notes))
    if any(item in hard_failures for item in integrity_warnings):
        integrity_status = "fail"
    elif integrity_warnings:
        integrity_status = "warn"
    else:
        integrity_status = "ok"
    return DataCoverageReport(
        symbol=str(symbol or "").upper(),
        market_type=str(market_type or "").lower(),
        interval=str(interval),
        source_scope=scope,
        expected_interval_ms=int(expected_ms),
        integrity_status=integrity_status,
        integrity_warnings=integrity_warnings,
        truth_basis=(
            "prices_from_timestamped_closed_candles",
            "coverage_measured_from_candle_close_time",
            "execution_results_are_simulated_not_exchange_fills",
        ),
        full_history_requested=full_requested,
        full_available_history_used=full_used,
        candles_available=len(available_candles),
        candles_used=len(used_candles),
        rows_used=int(rows_used),
        requested_start_ms=requested_start_ms,
        requested_end_ms=requested_end_ms,
        available_start_ms=available_start,
        available_end_ms=available_end,
        used_start_ms=used_start,
        used_end_ms=used_end,
        available_start_utc=iso_utc(available_start),
        available_end_utc=iso_utc(available_end),
        used_start_utc=iso_utc(used_start),
        used_end_utc=iso_utc(used_end),
        used_duration_days=duration_ms / _MS_PER_DAY,
        used_duration_years=duration_ms / _MS_PER_YEAR,
        gap_count=int(gap_count),
        largest_gap_ms=int(largest_gap_ms),
        largest_gap_intervals=float(largest_gap_intervals),
        coverage_ratio=float(coverage_ratio),
        notes=tuple(notes),
    )
