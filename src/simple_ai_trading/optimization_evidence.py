"""Reproducible optimization-round evidence and graph-data generation."""

from __future__ import annotations

import csv
import copy
import gc
import gzip
import hashlib
import json
import math
import statistics
from bisect import bisect_left, insort
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .advanced_model import (
    AdvancedFeatureConfig,
    advanced_feature_dimension,
    default_config_for,
    make_advanced_rows,
    train_advanced,
)
from .alpha_search import (
    DEFAULT_RULE_ALPHA_MAX_CANDIDATES,
    optimize_rule_alpha_model_zoo,
    summarize_rule_alpha_candidate_distribution,
    summarize_rule_alpha_trade_path,
)
from .api import BinanceAPIError, BinanceClient, Candle
from .assets import MAX_AUTONOMOUS_LEVERAGE, is_supported_major_symbol, major_symbols_for_quote
from .backtest import (
    BacktestResult,
    calibrate_threshold_for_backtest,
    closed_trades_per_day,
    run_backtest,
)
from .compute import resolve_backend
from .data_coverage import describe_candle_coverage, iso_utc
from .data_downloader import MarketDataSyncConfig, sync_market_data
from .execution_simulation import SymbolExecutionProfile
from .hybrid_models import optimize_hybrid_model_zoo
from .intervals import interval_milliseconds, validate_interval
from .market_edge import build_market_edge_report
from .market_store import MarketDataStore
from .market_universe import (
    _exchange_symbol_map,
    _looks_price_pegged,
    _looks_structurally_dangerous,
    _safe_float,
    _safe_int,
    _score_liquidity,
    _spread_bps,
)
from .model import calibrate_probability_temperature
from .model import effective_training_backend_name
from .objective import ObjectiveSpec, get_objective
from .performance_charts import EquityPoint
from .risk_controls import stop_loss_sized_notional_pct
from .storage import write_json_atomic
from .strategy_overrides import apply_model_strategy_overrides, strategy_overrides_from_config
from .types import StrategyConfig


@dataclass(frozen=True)
class EvidencePaths:
    output_dir: Path
    market_db_path: Path
    docs_dir: Path
    docs_data_dir: Path
    docs_charts_dir: Path
    report_path: Path
    data_health_path: Path
    status_path: Path
    progress_csv_path: Path
    metrics_csv_path: Path
    candidate_diagnostics_csv_path: Path
    candidate_diagnostics_json_path: Path
    timeline_csv_path: Path


@dataclass(frozen=True)
class SelectedSymbol:
    rank: int
    symbol: str
    quote_volume: float
    trade_count: int
    spread_bps: float
    liquidity_score: float
    selection_score: float
    strict_default_eligible: bool
    tier: str
    reasons: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestEvidence:
    round_id: str
    symbol: str
    objective: str
    risk_level: str
    leverage: float
    effective_leverage: float
    leverage_applies: bool
    risk_per_trade: float
    max_position_pct: float
    max_drawdown_limit_pct: float
    accepted: bool
    reason: str | None
    start_utc: str | None
    end_utc: str | None
    duration_years: float
    candles: int
    rows: int
    training_rows: int
    training_positive_rate_pct: float
    model_candidate_count: int
    model_selected_candidate: str
    model_selection_score: float | None
    candidate_diagnostics_path: str
    model_training_backend_kind: str
    model_training_backend_device: str
    probability_calibration_backend_kind: str
    probability_calibration_backend_device: str
    threshold_source: str | None
    threshold_calibration_score: float | None
    threshold_calibration_pnl: float | None
    threshold_calibration_trades: int
    threshold_diagnostic_best_threshold: float | None
    threshold_diagnostic_best_score: float | None
    threshold_diagnostic_best_pnl: float | None
    threshold_diagnostic_best_trades: int
    decision_threshold: float | None
    round_selection_gate_passed: bool
    round_selection_reject_reason: str | None
    model_quality_warnings: str
    meta_label_policy_reason: str | None
    starting_cash: float
    ending_cash: float
    realized_pnl: float
    roi_pct: float
    buy_hold_pnl: float
    buy_hold_roi_pct: float
    edge_vs_buy_hold: float
    market_edge_pct: float
    max_drawdown_pct: float
    trades: int
    closed_trades: int
    win_rate_pct: float
    total_fees: float
    profit_factor: float
    expectancy: float
    avg_trade_return_pct: float
    max_consecutive_losses: int
    low_liquidity_sample_rate_pct: float
    weekend_sample_rate_pct: float
    scoring_backend_kind: str
    scoring_backend_device: str
    chart_path: str
    timeline_csv_path: str
    stopped_by_liquidation: bool = False
    liquidation_events: int = 0
    liquidation_loss: float = 0.0
    long_decision_threshold: float | None = None
    short_decision_threshold: float | None = None
    threshold_diagnostic_best_long_threshold: float | None = None
    threshold_diagnostic_best_short_threshold: float | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def critical_round_analysis(metrics: Sequence[BacktestEvidence]) -> dict[str, object]:
    """Return a fail-closed interpretation of an optimization round.

    A round is not useful trading evidence merely because it completed. It must
    actually trade, produce accepted symbols, and show positive net outcomes.
    """

    symbol_count = len(metrics)
    accepted = [metric for metric in metrics if metric.accepted]
    total_trades = sum(int(metric.trades) for metric in metrics)
    total_closed_trades = sum(int(metric.closed_trades) for metric in metrics)
    profitable_symbols = [metric for metric in metrics if float(metric.realized_pnl) > 0.0]
    positive_roi_symbols = [metric for metric in metrics if float(metric.roi_pct) > 0.0]
    zero_trade_symbols = [metric.symbol for metric in metrics if int(metric.closed_trades) <= 0]
    nonpositive_roi_symbols = [metric.symbol for metric in metrics if float(metric.roi_pct) <= 0.0]
    negative_roi_symbols = [metric.symbol for metric in metrics if float(metric.roi_pct) < 0.0]
    liquidation_symbols = [
        metric.symbol
        for metric in metrics
        if bool(getattr(metric, "stopped_by_liquidation", False)) or int(getattr(metric, "liquidation_events", 0)) > 0
    ]
    selection_gate_failed_symbols = [
        metric.symbol for metric in metrics if not bool(metric.round_selection_gate_passed)
    ]
    rejected_diagnostic_trade_symbols = [
        metric.symbol
        for metric in metrics
        if not bool(metric.round_selection_gate_passed)
        and int(metric.threshold_diagnostic_best_trades) > 0
    ]
    failures: list[str] = []
    warnings: list[str] = []
    if symbol_count <= 0:
        failures.append("no_symbols_completed")
    if not accepted:
        failures.append("no_accepted_symbols")
    if total_closed_trades <= 0:
        failures.append("no_closed_trades")
    if symbol_count > 0 and len(zero_trade_symbols) == symbol_count:
        failures.append("all_symbols_zero_closed_trades")
    if symbol_count > 0 and len(nonpositive_roi_symbols) == symbol_count:
        failures.append("all_symbols_nonpositive_roi")
    if not profitable_symbols:
        failures.append("no_profitable_symbols")
    if liquidation_symbols:
        failures.append("liquidation_events_detected")
    if negative_roi_symbols:
        warnings.append("some_symbols_negative_roi")
    if zero_trade_symbols:
        warnings.append("some_symbols_zero_closed_trades")
    if liquidation_symbols:
        warnings.append("liquidated_symbols_present")
    if selection_gate_failed_symbols:
        warnings.append("some_symbols_failed_selection_gate")
    if rejected_diagnostic_trade_symbols:
        warnings.append("rejected_symbols_have_trade_diagnostics")
    if failures:
        verdict = "fail"
        if total_closed_trades <= 0:
            interpretation = (
                "invalid_no_trade_abstention: strategy ROI is flat because no holdout trades closed; "
                "this is not evidence of profitability even when the passive baseline lost money."
            )
        else:
            interpretation = "failed_optimization_round: completed backtests did not satisfy trading evidence gates."
    else:
        verdict = "pass"
        interpretation = "promotion_candidate: round has accepted symbols, closed trades, and positive net outcomes."
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "failures": failures,
        "warnings": warnings,
        "symbol_count": symbol_count,
        "accepted_symbol_count": len(accepted),
        "profitable_symbol_count": len(profitable_symbols),
        "positive_roi_symbol_count": len(positive_roi_symbols),
        "zero_trade_symbol_count": len(zero_trade_symbols),
        "nonpositive_roi_symbol_count": len(nonpositive_roi_symbols),
        "negative_roi_symbol_count": len(negative_roi_symbols),
        "selection_gate_failed_symbol_count": len(selection_gate_failed_symbols),
        "rejected_diagnostic_trade_symbol_count": len(rejected_diagnostic_trade_symbols),
        "total_trades": total_trades,
        "total_closed_trades": total_closed_trades,
        "total_threshold_diagnostic_best_trades": sum(
            int(metric.threshold_diagnostic_best_trades) for metric in metrics
        ),
        "mean_roi_pct": statistics.mean([metric.roi_pct for metric in metrics]) if metrics else 0.0,
        "median_roi_pct": statistics.median([metric.roi_pct for metric in metrics]) if metrics else 0.0,
        "mean_baseline_roi_pct": statistics.mean([metric.buy_hold_roi_pct for metric in metrics]) if metrics else 0.0,
        "zero_trade_symbols": zero_trade_symbols,
        "nonpositive_roi_symbols": nonpositive_roi_symbols,
        "negative_roi_symbols": negative_roi_symbols,
        "liquidation_symbols": liquidation_symbols,
        "total_liquidation_events": sum(int(getattr(metric, "liquidation_events", 0)) for metric in metrics),
        "selection_gate_failed_symbols": selection_gate_failed_symbols,
        "rejected_diagnostic_trade_symbols": rejected_diagnostic_trade_symbols,
    }


@dataclass(frozen=True)
class RoundModelCandidate:
    name: str
    feature_cfg: AdvancedFeatureConfig
    epochs: int
    learning_rate: float
    l2_penalty: float
    signal_threshold: float
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    cooldown_multiplier: float = 1.0
    min_position_hold_bars: int = 0
    flat_signal_exit_grace_bars: int = 0
    focal_gamma: float = 0.0


@dataclass(frozen=True)
class RoundModelCandidateResult:
    candidate: RoundModelCandidate
    score: float
    model: object
    report: object
    selection_result: BacktestResult
    selection_reject_reason: str | None
    training_rows: list[object]
    selection_rows: list[object]
    rows: list[object]
    validation_rows: list[object]


def make_evidence_paths(
    round_id: str,
    *,
    data_root: Path = Path("data/optimization"),
    docs_root: Path = Path("docs/optimization"),
    market_db_path: Path = Path("data/market_data.sqlite"),
) -> EvidencePaths:
    safe_round = str(round_id).strip().lower().replace(" ", "-")
    output_dir = data_root / safe_round
    docs_dir = docs_root / safe_round
    docs_data_dir = docs_dir / "data"
    docs_charts_dir = docs_dir / "charts"
    return EvidencePaths(
        output_dir=output_dir,
        market_db_path=market_db_path,
        docs_dir=docs_dir,
        docs_data_dir=docs_data_dir,
        docs_charts_dir=docs_charts_dir,
        report_path=docs_data_dir / "report.json",
        data_health_path=docs_data_dir / "data-health.json",
        status_path=docs_data_dir / "round-status.json",
        progress_csv_path=docs_data_dir / "round-progress.csv",
        metrics_csv_path=docs_data_dir / "backtest-metrics.csv",
        candidate_diagnostics_csv_path=docs_data_dir / "candidate-diagnostics.csv",
        candidate_diagnostics_json_path=docs_data_dir / "candidate-diagnostics.json",
        timeline_csv_path=docs_data_dir / "portfolio-timeline.csv.gz",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _promotion_min_rows(min_data_years: float, interval: str) -> int:
    seconds = max(0.0, _finite(min_data_years, 0.0)) * 365.25 * 24 * 60 * 60
    interval_ms = max(1, interval_milliseconds(interval))
    return int(math.floor(seconds * 1000.0 / interval_ms))


def _normalize_symbol_list(symbols: Sequence[str] | None) -> list[str]:
    return [str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()]


def _promotion_required_symbols(quote_asset: str) -> list[str]:
    return list(major_symbols_for_quote(quote_asset))


def _validate_promotion_symbol_scope(symbols: Sequence[str], quote_asset: str) -> None:
    required = _promotion_required_symbols(quote_asset)
    requested = list(dict.fromkeys(_normalize_symbol_list(symbols)))
    if sorted(requested) != sorted(required):
        raise ValueError(
            "promotion_grade_requires_exact_btc_eth_sol_scope: "
            f"required={','.join(required)} requested={','.join(requested) or 'none'}"
        )


def promotion_grade_contract(
    *,
    market_type: str,
    quote_asset: str,
    interval: str,
    selected_symbols: Sequence[str],
    data_health: Sequence[Mapping[str, object]],
    critical_analysis: Mapping[str, object],
    require_prefilled_data: bool,
    require_verified_checksum: bool,
    min_data_rows: int,
    min_data_years: float,
    min_coverage_ratio: float,
    max_gap_count: int,
    require_gpu: bool,
    backend_kind: str,
) -> dict[str, object]:
    """Return the fail-closed contract for publishable day-trading evidence."""

    required = _promotion_required_symbols(quote_asset)
    selected = list(dict.fromkeys(_normalize_symbol_list(selected_symbols)))
    reasons: list[str] = []
    if str(interval) != "1s":
        reasons.append("interval_not_1s")
    if sorted(selected) != sorted(required):
        reasons.append("symbol_scope_not_exact_btc_eth_sol")
    if not require_prefilled_data:
        reasons.append("network_backfill_not_disabled")
    if not require_verified_checksum:
        reasons.append("verified_archive_checksum_not_required")
    if min_data_rows < _promotion_min_rows(min_data_years, interval):
        reasons.append("min_data_rows_below_promotion_year_requirement")
    if min_coverage_ratio < 0.995:
        reasons.append("coverage_ratio_gate_below_99_5_percent")
    if max_gap_count != 0:
        reasons.append("gap_gate_allows_missing_seconds")
    if require_gpu and str(backend_kind).lower() == "cpu":
        reasons.append("gpu_required_but_backend_cpu")

    health_by_symbol = {str(item.get("symbol") or "").upper(): item for item in data_health if isinstance(item, Mapping)}
    for symbol in required:
        health = health_by_symbol.get(symbol)
        if health is None:
            reasons.append(f"missing_data_health:{symbol}")
            continue
        if health.get("status") != "ok":
            reasons.append(f"data_health_failed:{symbol}")
        if str(health.get("interval") or "") != "1s":
            reasons.append(f"data_health_interval_not_1s:{symbol}")
        if str(health.get("market_type") or "").lower() != str(market_type or "").lower():
            reasons.append(f"data_health_market_mismatch:{symbol}")
        if int(health.get("gap_count") or 0) != 0:
            reasons.append(f"data_health_gaps:{symbol}")
        if float(health.get("coverage_ratio") or 0.0) < min_coverage_ratio:
            reasons.append(f"data_health_coverage_below_gate:{symbol}")
        if float(health.get("span_years") or 0.0) < max(0.0, _finite(min_data_years, 0.0)):
            reasons.append(f"data_health_span_years_below_gate:{symbol}")
        checksum_counts = health.get("checksum_status_counts")
        if not isinstance(checksum_counts, Mapping) or int(checksum_counts.get("verified", 0) or 0) <= 0:
            reasons.append(f"data_health_missing_verified_checksum:{symbol}")

    if critical_analysis.get("verdict") != "pass":
        reasons.append("critical_analysis_not_pass")
    status = "pass" if not reasons else "block"
    return {
        "status": status,
        "reasons": sorted(dict.fromkeys(reasons)),
        "required_symbols": required,
        "selected_symbols": selected,
        "required_interval": "1s",
        "market_type": str(market_type or "").lower(),
        "quote_asset": str(quote_asset or "").upper(),
        "require_prefilled_data": bool(require_prefilled_data),
        "require_verified_checksum": bool(require_verified_checksum),
        "min_data_rows": int(min_data_rows),
        "min_data_years": float(max(0.0, _finite(min_data_years, 0.0))),
        "min_coverage_ratio": float(min_coverage_ratio),
        "max_gap_count": int(max_gap_count),
        "require_gpu": bool(require_gpu),
        "backend_kind": str(backend_kind),
        "critical_verdict": str(critical_analysis.get("verdict") or "unknown"),
    }


def effective_leverage_for_market(strategy: StrategyConfig, market_type: str) -> float:
    """Return the leverage that can actually affect fills for the market type."""

    if str(market_type).lower() != "futures":
        return 1.0
    return max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, _finite(strategy.leverage, 1.0)))


def parse_evidence_timestamp_ms(value: object, *, end_of_day: bool = False) -> int | None:
    """Parse an evidence-window boundary as UTC milliseconds."""

    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    date_only = len(text) == 10 and text[4] == "-" and text[7] == "-"
    try:
        if date_only:
            parsed = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if end_of_day:
                parsed = parsed + timedelta(days=1) - timedelta(milliseconds=1)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid evidence timestamp: {text}") from exc
    return int(parsed.timestamp() * 1000)


def _archive_period_bounds_ms(period: object) -> tuple[int, int] | None:
    text = str(period or "").strip()
    try:
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            start = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = start + timedelta(days=1) - timedelta(milliseconds=1)
            return int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        if len(text) == 7 and text[4] == "-":
            start = datetime.strptime(text, "%Y-%m").replace(tzinfo=timezone.utc)
            if start.month == 12:
                next_month = start.replace(year=start.year + 1, month=1)
            else:
                next_month = start.replace(month=start.month + 1)
            end = next_month - timedelta(milliseconds=1)
            return int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    except ValueError:
        return None
    return None


def _filter_archives_for_window(
    archives: Sequence[object],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> list[object]:
    if start_ms is None and end_ms is None:
        return list(archives)
    lower = -2**63 if start_ms is None else int(start_ms)
    upper = 2**63 - 1 if end_ms is None else int(end_ms)
    filtered: list[object] = []
    for archive in archives:
        bounds = _archive_period_bounds_ms(getattr(archive, "period", ""))
        if bounds is None:
            continue
        period_start, period_end = bounds
        if period_end >= lower and period_start <= upper:
            filtered.append(archive)
    return filtered


def fetch_full_history(
    client: BinanceClient,
    symbol: str,
    interval: str,
    *,
    db_path: Path,
    market_type: str = "spot",
    batch_size: int = 1000,
    allow_network_backfill: bool = True,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[Candle]:
    with MarketDataStore(db_path) as store:
        candles = store.fetch_candles(symbol, market_type, interval, start_ms=start_ms, end_ms=end_ms)
    if candles:
        return candles
    if not allow_network_backfill:
        raise ValueError(f"market database has no prefilled candles for {symbol} {market_type} {interval}")

    result = sync_market_data(
        client,
        MarketDataSyncConfig(
            symbol=symbol,
            interval=interval,
            market_type=market_type,
            db_path=db_path,
            rows=0,
            batch_size=batch_size,
            include_futures_metrics=False,
            full_history=True,
        ),
    )
    if result.status == "fail":
        raise BinanceAPIError("; ".join(result.errors) or f"failed to backfill {symbol} {interval}")
    with MarketDataStore(db_path) as store:
        candles = store.fetch_candles(symbol, market_type, interval, start_ms=start_ms, end_ms=end_ms)
        quality = store.coverage_quality(
            symbol,
            market_type,
            interval,
            interval_milliseconds(interval),
            start_ms=start_ms,
            end_ms=end_ms,
        )
    if not candles:
        raise ValueError(f"no candles available in market database for {symbol} {interval}")
    if quality.gap_count:
        raise ValueError(f"market database has {quality.gap_count} gaps for {symbol} {interval}")
    return candles


def _count_by(items: Sequence[object], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(getattr(item, attr, "") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def market_data_health_for_symbol(
    *,
    db_path: Path,
    symbol: str,
    market_type: str,
    interval: str,
    min_rows: int = 0,
    min_coverage_ratio: float = 0.995,
    max_gap_count: int = 0,
    require_verified_checksum: bool = False,
    min_span_years: float = 0.0,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> dict[str, object]:
    """Return a fail-closed health report for one optimization data series."""

    with MarketDataStore(db_path) as store:
        quality = store.coverage_quality(
            symbol,
            market_type,
            interval,
            interval_milliseconds(interval),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        archives = _filter_archives_for_window(
            store.archive_files(symbol=symbol, market_type=market_type, interval=interval),
            start_ms=start_ms,
            end_ms=end_ms,
        )
    archive_status_counts = _count_by(archives, "status")
    checksum_status_counts = _count_by(archives, "checksum_status")
    min_rows = max(0, int(min_rows))
    max_gap_count = max(0, int(max_gap_count))
    min_coverage_ratio = max(0.0, min(1.0, float(min_coverage_ratio)))
    min_span_years = max(0.0, _finite(min_span_years, 0.0))
    first_open_time = quality.coverage.first_open_time
    last_open_time = quality.coverage.last_open_time
    span_ms = (
        max(0, int(last_open_time) - int(first_open_time))
        if first_open_time is not None and last_open_time is not None
        else 0
    )
    span_days = span_ms / (24 * 60 * 60 * 1000)
    span_years = span_days / 365.25
    reasons: list[str] = []
    warnings: list[str] = []
    if quality.coverage.count < min_rows:
        reasons.append(f"rows_below_min:{quality.coverage.count}/{min_rows}")
    if span_years < min_span_years:
        reasons.append(f"span_years_below_min:{span_years:.6f}/{min_span_years:.6f}")
    if quality.gap_count > max_gap_count:
        reasons.append(f"gap_count_above_max:{quality.gap_count}/{max_gap_count}")
    if quality.coverage_ratio < min_coverage_ratio:
        reasons.append(f"coverage_ratio_below_min:{quality.coverage_ratio:.6f}/{min_coverage_ratio:.6f}")
    if checksum_status_counts.get("mismatch", 0) > 0:
        reasons.append(f"checksum_mismatches:{checksum_status_counts['mismatch']}")
    if require_verified_checksum and checksum_status_counts.get("verified", 0) <= 0:
        reasons.append("no_verified_archive_checksum")
    archive_error_count = archive_status_counts.get("error", 0)
    if archive_error_count > 0:
        superseded_by_verified_coverage = (
            quality.coverage.count >= min_rows
            and quality.gap_count <= max_gap_count
            and quality.coverage_ratio >= min_coverage_ratio
            and checksum_status_counts.get("mismatch", 0) <= 0
            and (not require_verified_checksum or checksum_status_counts.get("verified", 0) > 0)
        )
        if superseded_by_verified_coverage:
            warnings.append(f"superseded_archive_errors:{archive_error_count}")
        else:
            reasons.append(f"archive_errors:{archive_error_count}")
    return {
        "status": "ok" if not reasons else "block",
        "symbol": symbol.upper(),
        "market_type": market_type,
        "interval": interval,
        "rows": quality.coverage.count,
        "expected_rows": quality.expected_count,
        "first_open_time": quality.coverage.first_open_time,
        "last_open_time": quality.coverage.last_open_time,
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "requested_start_utc": iso_utc(start_ms),
        "requested_end_utc": iso_utc(end_ms),
        "span_days": span_days,
        "span_years": span_years,
        "coverage_ratio": quality.coverage_ratio,
        "gap_count": quality.gap_count,
        "archive_status_counts": archive_status_counts,
        "checksum_status_counts": checksum_status_counts,
        "reasons": reasons,
        "warnings": warnings,
    }


def select_top_liquidity_symbols(
    client: BinanceClient,
    strategy: StrategyConfig,
    *,
    quote_asset: str = "USDT",
    count: int = 50,
    max_scan: int = 1000,
    strict_only: bool = True,
) -> list[SelectedSymbol]:
    quote_asset = str(quote_asset or "USDT").upper()
    exchange_symbols = _exchange_symbol_map(client)
    tickers = {
        str(item.get("symbol") or "").upper(): item
        for item in client.get_all_tickers_24h()
        if isinstance(item, Mapping) and item.get("symbol")
    }
    books = {
        str(item.get("symbol") or "").upper(): item
        for item in client.get_all_book_tickers()
        if isinstance(item, Mapping) and item.get("symbol")
    }
    candidates: list[tuple[float, SelectedSymbol]] = []
    for symbol, symbol_info in exchange_symbols.items():
        if len(candidates) >= max(1, int(max_scan)):
            break
        if not symbol.endswith(quote_asset):
            continue
        if not is_supported_major_symbol(symbol, quote_asset):
            continue
        if str(symbol_info.get("status") or "") != "TRADING":
            continue
        if _looks_structurally_dangerous(symbol, quote_asset):
            continue
        ticker = tickers.get(symbol)
        book = books.get(symbol)
        if ticker is None or book is None or _looks_price_pegged(ticker):
            continue
        quote_volume = _safe_float(ticker.get("quoteVolume"))
        trade_count = _safe_int(ticker.get("count"))
        spread_bps = _spread_bps(book)
        relaxed_strategy = StrategyConfig(
            min_quote_volume_usdc=max(1_000_000.0, min(float(strategy.min_quote_volume_usdc), 10_000_000.0)),
            min_trade_count_24h=max(5_000, min(int(strategy.min_trade_count_24h), 10_000)),
            max_spread_bps=max(float(strategy.max_spread_bps), 20.0),
            min_liquidity_score=min(float(strategy.min_liquidity_score), 0.50),
        )
        liquidity_score = _score_liquidity(
            quote_volume=quote_volume,
            trade_count=trade_count,
            spread_bps=spread_bps,
            strategy=relaxed_strategy,
        )
        strict_reasons: list[str] = []
        if quote_volume < strategy.min_quote_volume_usdc:
            strict_reasons.append("quote_volume_below_default_live_gate")
        if trade_count < strategy.min_trade_count_24h:
            strict_reasons.append("trade_count_below_default_live_gate")
        if spread_bps > strategy.max_spread_bps:
            strict_reasons.append("spread_above_default_live_gate")
        if liquidity_score < relaxed_strategy.min_liquidity_score:
            strict_reasons.append("liquidity_score_below_research_gate")
        strict_eligible = not strict_reasons
        if strict_eligible:
            tier = "strict-live-eligible-at-selection"
        elif quote_volume >= 10_000_000.0 and trade_count >= 10_000 and spread_bps <= 20.0:
            tier = "research-high-liquidity"
        else:
            tier = "research-ranked"
        selection_score = (
            math.log10(max(1.0, quote_volume))
            + math.log10(max(1.0, float(trade_count))) * 0.60
            + liquidity_score * 2.0
            - min(3.0, max(0.0, spread_bps) / 20.0)
        )
        candidates.append((
            selection_score,
            SelectedSymbol(
                rank=0,
                symbol=symbol,
                quote_volume=float(quote_volume),
                trade_count=int(trade_count),
                spread_bps=float(spread_bps),
                liquidity_score=float(liquidity_score),
                selection_score=float(selection_score),
                strict_default_eligible=strict_eligible,
                tier=tier,
                reasons=tuple(strict_reasons),
            ),
        ))
    ranked_all = [item for _score, item in sorted(candidates, key=lambda row: row[0], reverse=True)]
    ranked = [item for item in ranked_all if item.strict_default_eligible] if strict_only else ranked_all
    return [
        SelectedSymbol(
            rank=index,
            symbol=item.symbol,
            quote_volume=item.quote_volume,
            trade_count=item.trade_count,
            spread_bps=item.spread_bps,
            liquidity_score=item.liquidity_score,
            selection_score=item.selection_score,
            strict_default_eligible=item.strict_default_eligible,
            tier=item.tier,
            reasons=item.reasons,
        )
        for index, item in enumerate(ranked[: max(1, int(count))], start=1)
    ]


def select_data_healthy_top_liquidity_symbols(
    client: BinanceClient,
    strategy: StrategyConfig,
    *,
    quote_asset: str = "USDT",
    count: int = 50,
    market_type: str,
    interval: str,
    db_path: Path,
    min_rows: int = 0,
    min_coverage_ratio: float = 0.995,
    max_gap_count: int = 0,
    require_verified_checksum: bool = False,
    min_span_years: float = 0.0,
    start_ms: int | None = None,
    end_ms: int | None = None,
    max_scan: int = 1000,
) -> tuple[list[SelectedSymbol], list[dict[str, object]]]:
    """Return live-ranked symbols that also pass local market-data health gates."""

    requested = max(1, int(count))
    candidate_count = max(1, int(max_scan))
    candidates = select_top_liquidity_symbols(
        client,
        strategy,
        quote_asset=quote_asset,
        count=candidate_count,
        max_scan=max_scan,
        strict_only=True,
    )
    selected: list[SelectedSymbol] = []
    health_rejections: list[dict[str, object]] = []
    for item in candidates:
        health = market_data_health_for_symbol(
            db_path=db_path,
            symbol=item.symbol,
            market_type=market_type,
            interval=interval,
            min_rows=min_rows,
            min_coverage_ratio=min_coverage_ratio,
            max_gap_count=max_gap_count,
            require_verified_checksum=require_verified_checksum,
            min_span_years=min_span_years,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if health.get("status") == "ok":
            selected.append(replace(item, rank=len(selected) + 1))
        else:
            health_rejections.append({
                "selection_rank": int(item.rank),
                "symbol": item.symbol,
                "tier": item.tier,
                "rows": int(health.get("rows") or 0),
                "span_years": float(health.get("span_years") or 0.0),
                "coverage_ratio": float(health.get("coverage_ratio") or 0.0),
                "gap_count": int(health.get("gap_count") or 0),
                "reasons": list(health.get("reasons") or []),
            })
        if len(selected) >= requested:
            break
    if len(selected) < requested:
        raise ValueError(
            "data_health_selection_shortfall: "
            f"selected {len(selected)}/{requested} symbols after scanning {len(candidates)} live-ranked candidates"
        )
    return selected, health_rejections


def select_named_symbols(
    client: BinanceClient,
    strategy: StrategyConfig,
    symbols: Sequence[str],
    *,
    quote_asset: str = "USDT",
) -> list[SelectedSymbol]:
    """Build selection metadata for an explicit operator-supplied symbol set."""

    quote_asset = str(quote_asset or "USDT").upper()
    requested = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not requested:
        return []
    exchange_symbols = _exchange_symbol_map(client)
    tickers = {
        str(item.get("symbol") or "").upper(): item
        for item in client.get_all_tickers_24h()
        if isinstance(item, Mapping) and item.get("symbol")
    }
    books = {
        str(item.get("symbol") or "").upper(): item
        for item in client.get_all_book_tickers()
        if isinstance(item, Mapping) and item.get("symbol")
    }
    selected: list[SelectedSymbol] = []
    for index, symbol in enumerate(dict.fromkeys(requested), start=1):
        symbol_info = exchange_symbols.get(symbol, {})
        ticker = tickers.get(symbol, {})
        book = books.get(symbol, {})
        quote_volume = _safe_float(ticker.get("quoteVolume")) if isinstance(ticker, Mapping) else 0.0
        trade_count = _safe_int(ticker.get("count")) if isinstance(ticker, Mapping) else 0
        spread_bps = _spread_bps(book if isinstance(book, Mapping) else {})
        liquidity_score = _score_liquidity(
            quote_volume=quote_volume,
            trade_count=trade_count,
            spread_bps=spread_bps,
            strategy=strategy,
        )
        reasons: list[str] = []
        if not symbol.endswith(quote_asset):
            reasons.append("quote_asset_mismatch")
        if not is_supported_major_symbol(symbol, quote_asset):
            reasons.append("unsupported_non_major_asset")
        if not symbol_info:
            reasons.append("missing_exchange_metadata")
        elif str(symbol_info.get("status") or "") != "TRADING":
            reasons.append("not_trading")
        if _looks_structurally_dangerous(symbol, quote_asset):
            reasons.append("leveraged_or_inverse_token_pattern")
        if isinstance(ticker, Mapping) and ticker and _looks_price_pegged(ticker):
            reasons.append("stable_or_pegged_pair_pattern")
        if quote_volume < strategy.min_quote_volume_usdc:
            reasons.append("quote_volume_below_default_live_gate")
        if trade_count < strategy.min_trade_count_24h:
            reasons.append("trade_count_below_default_live_gate")
        if spread_bps > strategy.max_spread_bps:
            reasons.append("spread_above_default_live_gate")
        selected.append(
            SelectedSymbol(
                rank=index,
                symbol=symbol,
                quote_volume=float(quote_volume),
                trade_count=int(trade_count),
                spread_bps=float(spread_bps),
                liquidity_score=float(liquidity_score),
                selection_score=float(liquidity_score),
                strict_default_eligible=not reasons,
                tier="explicit-symbol",
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return selected


def _split_train_validation(rows: Sequence[object], validation_fraction: float = 0.25) -> tuple[list[object], list[object]]:
    row_list = list(rows)
    if len(row_list) < 4:
        return row_list, []
    validation_size = max(1, int(round(len(row_list) * validation_fraction)))
    validation_size = min(validation_size, max(1, len(row_list) // 2))
    return row_list[:-validation_size], row_list[-validation_size:]


def strategy_with_objective_defaults(strategy: StrategyConfig, objective: ObjectiveSpec) -> StrategyConfig:
    """Apply the objective's trading defaults while preserving unrelated safeguards."""

    training = objective.training
    if training is None:
        return StrategyConfig(**{**strategy.asdict(), "risk_level": objective.name})
    return StrategyConfig(
        **{
            **strategy.asdict(),
            "risk_level": objective.name,
            "leverage": float(training.leverage),
            "signal_threshold": float(training.signal_threshold),
            "stop_loss_pct": float(training.stop_loss_pct),
            "take_profit_pct": float(training.take_profit_pct),
            "risk_per_trade": float(training.risk_per_trade),
            "max_position_pct": float(training.max_position_pct),
            "max_trades_per_day": int(training.max_trades_per_day),
            "cooldown_minutes": int(training.cooldown_minutes),
            "training_epochs": int(training.epochs),
        }
    )


def _strategy_for_round_candidate(strategy: StrategyConfig, candidate: RoundModelCandidate) -> StrategyConfig:
    return replace(
        strategy,
        stop_loss_pct=max(0.001, float(strategy.stop_loss_pct) * max(0.05, float(candidate.stop_loss_multiplier))),
        take_profit_pct=max(0.001, float(strategy.take_profit_pct) * max(0.05, float(candidate.take_profit_multiplier))),
        cooldown_minutes=max(0, int(round(float(strategy.cooldown_minutes) * max(0.0, float(candidate.cooldown_multiplier))))),
        min_position_hold_bars=max(0, int(candidate.min_position_hold_bars)),
        flat_signal_exit_grace_bars=max(0, int(candidate.flat_signal_exit_grace_bars)),
    )


def _baseline_equity_series(rows: Sequence[object], starting_cash: float, cfg: StrategyConfig, *, market_type: str) -> list[dict[str, float | int]]:
    if not rows:
        return []
    first = _finite(getattr(rows[0], "close", 0.0))
    if first <= 0.0:
        return []
    # Use the same conservative risk notional convention as backtest._buy_hold_pnl.
    notional_pct = stop_loss_sized_notional_pct(cfg, market_type)
    baseline_cash = float(starting_cash)
    notional = baseline_cash * notional_pct
    fee_rate = max(0.0, float(cfg.taker_fee_bps)) / 10_000.0
    entry_cost = notional * (1.0 + fee_rate)
    idle_cash = baseline_cash - entry_cost
    qty = notional / first
    points: list[dict[str, float | int]] = []
    peak = baseline_cash
    for row in rows:
        price = _finite(getattr(row, "close", 0.0))
        equity = idle_cash + qty * price
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0.0 else 0.0
        points.append({"timestamp": int(getattr(row, "timestamp", 0)), "equity": float(equity), "drawdown": float(dd)})
    return points


def _liquidity_clock_bucket(timestamp_ms: int, bucket_minutes: int = 15) -> tuple[int, int, int]:
    dt = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
    bucket = max(1, min(60, int(bucket_minutes)))
    return dt.weekday(), dt.hour, dt.minute // bucket


def _median_sorted(values: Sequence[float]) -> float:
    count = len(values)
    if count <= 0:
        return 0.0
    mid = count // 2
    if count % 2:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _discard_sorted_value(values: list[float], value: float) -> None:
    index = bisect_left(values, value)
    if index < len(values):
        values.pop(index)


def _rolling_liquidity_flags(
    candles: Sequence[Candle],
    *,
    window: int = 96,
    timestamps: Iterable[int] | None = None,
) -> dict[int, dict[str, float | int | bool | str]]:
    flags: dict[int, dict[str, float | int | bool | str]] = {}
    wanted: set[int] | None = None
    max_wanted: int | None = None
    if timestamps is not None:
        wanted = {int(timestamp) for timestamp in timestamps if int(timestamp) > 0}
        if not wanted:
            return flags
        max_wanted = max(wanted)

    lookback = max(1, int(window))
    sorted_volumes: list[float] = []
    sorted_trades: list[float] = []
    prior_window: deque[tuple[tuple[int, int, int], float, float]] = deque()
    bucket_volumes: defaultdict[tuple[int, int, int], list[float]] = defaultdict(list)
    bucket_trades: defaultdict[tuple[int, int, int], list[float]] = defaultdict(list)

    for candle in candles:
        close_time = int(candle.close_time)
        if max_wanted is not None and close_time > max_wanted:
            break
        volume = max(0.0, float(candle.quote_volume))
        trades = float(max(0, int(candle.trade_count)))
        median_volume = _median_sorted(sorted_volumes)
        median_trades = _median_sorted(sorted_trades)
        dt = datetime.fromtimestamp(close_time / 1000.0, tz=timezone.utc)
        bucket = (dt.weekday(), dt.hour, dt.minute // 15)
        same_bucket_volumes = bucket_volumes.get(bucket, [])
        same_bucket_trades = bucket_trades.get(bucket, [])
        bucket_median_volume = _median_sorted(same_bucket_volumes) if len(same_bucket_volumes) >= 8 else 0.0
        bucket_median_trades = _median_sorted(same_bucket_trades) if len(same_bucket_trades) >= 8 else 0.0
        if wanted is None or close_time in wanted:
            low_volume = bool(median_volume > 0 and volume < median_volume * 0.35)
            low_trades = bool(median_trades > 0 and trades < median_trades * 0.35)
            low_bucket_volume = bool(bucket_median_volume > 0 and volume < bucket_median_volume * 0.45)
            low_bucket_trades = bool(bucket_median_trades > 0 and trades < bucket_median_trades * 0.45)
            flags[close_time] = {
                "quote_volume": float(volume),
                "trade_count": int(trades),
                "rolling_quote_volume_median": float(median_volume),
                "rolling_trade_count_median": float(median_trades),
                "clock_bucket": f"{bucket[0]}:{bucket[1]:02d}:{bucket[2]:02d}",
                "clock_bucket_quote_volume_median": float(bucket_median_volume),
                "clock_bucket_trade_count_median": float(bucket_median_trades),
                "data_probed_low_session_flag": bool(low_bucket_volume or low_bucket_trades),
                "low_liquidity_flag": bool(low_volume or low_trades or low_bucket_volume or low_bucket_trades),
                "weekend_flag": bool(dt.weekday() >= 5),
                "utc_hour": int(dt.hour),
                "utc_weekday": int(dt.weekday()),
            }
        prior_window.append((bucket, volume, trades))
        insort(sorted_volumes, volume)
        insort(sorted_trades, trades)
        insort(bucket_volumes[bucket], volume)
        insort(bucket_trades[bucket], trades)
        if len(prior_window) > lookback:
            old_bucket, old_volume, old_trades = prior_window.popleft()
            _discard_sorted_value(sorted_volumes, old_volume)
            _discard_sorted_value(sorted_trades, old_trades)
            old_bucket_volumes = bucket_volumes.get(old_bucket, [])
            old_bucket_trades = bucket_trades.get(old_bucket, [])
            _discard_sorted_value(old_bucket_volumes, old_volume)
            _discard_sorted_value(old_bucket_trades, old_trades)
            if not old_bucket_volumes:
                bucket_volumes.pop(old_bucket, None)
            if not old_bucket_trades:
                bucket_trades.pop(old_bucket, None)
    return flags


def _result_points(result: BacktestResult) -> list[EquityPoint]:
    points: list[EquityPoint] = []
    for index, point in enumerate(getattr(result, "equity_curve", ()) or ()):
        if not isinstance(point, Mapping) or "equity" not in point:
            continue
        points.append(EquityPoint(
            index=index,
            equity=_finite(point.get("equity")),
            drawdown=max(0.0, _finite(point.get("drawdown"))),
            timestamp_ms=int(point.get("timestamp")) if point.get("timestamp") is not None else None,
        ))
    return points


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_writer(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _open_text_writer(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def _open_text_reader(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _artifact_path(raw: str | Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else Path.cwd() / path


def _csv_shape(path: Path) -> tuple[int, tuple[str, ...]]:
    try:
        with _open_text_reader(path) as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            rows = sum(1 for _row in reader)
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, csv.Error):
        return 0, ()
    return max(0, int(rows)), tuple(str(column) for column in (header or ()))


def _artifact_integrity(raw: str | Path) -> dict[str, object]:
    normalized = str(raw).replace("\\", "/")
    path = _artifact_path(raw)
    payload = path.read_bytes()
    entry: dict[str, object] = {
        "path": normalized,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    lower = normalized.lower()
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        rows, columns = _csv_shape(path)
        entry["row_count"] = rows
        entry["columns"] = list(columns)
    return entry


def _artifact_integrity_manifest(
    tracked_artifacts: Sequence[str],
    *,
    report_path: Path,
) -> list[dict[str, object]]:
    report = str(report_path).replace("\\", "/")
    manifest: list[dict[str, object]] = []
    for raw in sorted(dict.fromkeys(str(item).replace("\\", "/") for item in tracked_artifacts)):
        if raw == report:
            continue
        path = _artifact_path(raw)
        if not path.exists() or not path.is_file():
            manifest.append({"path": raw, "missing": True})
            continue
        manifest.append(_artifact_integrity(raw))
    return manifest


def _write_round_status(paths: EvidencePaths, **payload: object) -> None:
    status = {
        "generated_at_utc": _utc_now(),
        "round_id": paths.docs_dir.name,
        **payload,
    }
    write_json_atomic(paths.status_path, status, indent=2, sort_keys=True)


def _require_non_cpu_backend(kind: object, reason: object, stage: str) -> None:
    if str(kind or "").lower() != "cpu":
        return
    suffix = f": {reason}" if reason else ""
    raise RuntimeError(f"gpu_required_but_{stage}_fell_back_to_cpu{suffix}")


def _decimate_equity_points(points: Sequence[EquityPoint], *, max_points: int = 6000) -> list[EquityPoint]:
    """Return a deterministic visual summary while preserving raw CSV evidence."""

    ordered = list(points)
    limit = max(16, int(max_points))
    if len(ordered) <= limit:
        return ordered
    bucket_size = max(1, math.ceil(len(ordered) / max(1, limit // 4)))
    keep: dict[int, EquityPoint] = {0: ordered[0], len(ordered) - 1: ordered[-1]}
    for start in range(0, len(ordered), bucket_size):
        bucket = ordered[start:start + bucket_size]
        if not bucket:
            continue
        candidates = (
            bucket[0],
            bucket[-1],
            min(bucket, key=lambda point: point.equity),
            max(bucket, key=lambda point: point.equity),
            max(bucket, key=lambda point: point.drawdown),
        )
        for point in candidates:
            keep[start + bucket.index(point)] = point
    return [point for _index, point in sorted(keep.items())]


def render_comparison_svg(
    strategy_points: Sequence[EquityPoint],
    baseline_points: Sequence[EquityPoint],
    *,
    title: str,
    width: int = 1120,
    height: int = 660,
) -> str:
    raw_points = list(strategy_points)
    raw_baseline = list(baseline_points)
    if not raw_points:
        raw_points = [EquityPoint(0, 0.0, 0.0)]
    if not raw_baseline:
        raw_baseline = [EquityPoint(point.index, raw_points[0].equity, 0.0, point.timestamp_ms) for point in raw_points]
    left, right, top, bottom = 72, 34, 62, 82
    chart_w = width - left - right
    chart_h = height - top - bottom
    all_equity = [point.equity for point in raw_points] + [point.equity for point in raw_baseline]
    min_equity = min(all_equity)
    max_equity = max(all_equity)
    max_drawdown = max(0.01, *(point.drawdown for point in raw_points))
    points = _decimate_equity_points(raw_points)
    baseline = _decimate_equity_points(raw_baseline)
    max_index = max(1, *(point.index for point in [*points, *baseline]))

    def sx(index: int) -> float:
        return left + (index / max_index) * chart_w

    def sy(value: float) -> float:
        if max_equity <= min_equity:
            return top + chart_h / 2.0
        return top + ((max_equity - value) / (max_equity - min_equity)) * chart_h

    def dy(value: float) -> float:
        return top + chart_h - (value / max_drawdown) * chart_h * 0.32

    strategy_poly = " ".join(f"{sx(point.index):.2f},{sy(point.equity):.2f}" for point in points)
    baseline_poly = " ".join(f"{sx(point.index):.2f},{sy(point.equity):.2f}" for point in baseline)
    drawdown_poly = " ".join(f"{sx(point.index):.2f},{dy(point.drawdown):.2f}" for point in points)
    timestamps = [point.timestamp_ms for point in raw_points if point.timestamp_ms is not None]
    start = iso_utc(min(timestamps))[:10] if timestamps else "sample"
    end = iso_utc(max(timestamps))[:10] if timestamps else "index"
    chart_note = f"Rendered {len(points):,}/{len(raw_points):,} strategy points; full-resolution graph data is in CSV."
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="{left}" y="36" font-family="Segoe UI, Arial, sans-serif" font-size="24" fill="#111827">{title}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#94a3b8"/>
  <line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#94a3b8"/>
  <polyline points="{baseline_poly}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="7 5"/>
  <polyline points="{drawdown_poly}" fill="none" stroke="#dc2626" stroke-width="2" opacity="0.80"/>
  <polyline points="{strategy_poly}" fill="none" stroke="#0f766e" stroke-width="3"/>
  <rect x="{left}" y="{height - 64}" width="16" height="4" fill="#0f766e"/>
  <text x="{left + 24}" y="{height - 58}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">strategy equity</text>
  <line x1="{left + 170}" y1="{height - 62}" x2="{left + 190}" y2="{height - 62}" stroke="#64748b" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="{left + 198}" y="{height - 58}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">passive baseline</text>
  <line x1="{left + 360}" y1="{height - 62}" x2="{left + 382}" y2="{height - 62}" stroke="#dc2626" stroke-width="2"/>
  <text x="{left + 390}" y="{height - 58}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">drawdown</text>
  <text x="{left}" y="{height - 28}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">UTC span: {start} to {end}. {chart_note}</text>
  <text x="{left}" y="{top + chart_h + 18}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">{start}</text>
  <text x="{left + chart_w - 84}" y="{top + chart_h + 18}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">{end}</text>
</svg>
"""


def _update_portfolio_aggregate(aggregate: dict[int, dict[str, float]], row: Mapping[str, object]) -> None:
    try:
        timestamp = int(row.get("timestamp_ms", 0))
    except (TypeError, ValueError):
        return
    if timestamp <= 0:
        return
    item = aggregate.setdefault(
        timestamp,
        {
            "strategy_sum": 0.0,
            "baseline_sum": 0.0,
            "drawdown_sum": 0.0,
            "count": 0.0,
            "low_liquidity_count": 0.0,
        },
    )
    item["strategy_sum"] += _finite(row.get("strategy_equity"))
    item["baseline_sum"] += _finite(row.get("baseline_equity"))
    item["drawdown_sum"] += _finite(row.get("strategy_drawdown"))
    item["count"] += 1.0
    if str(row.get("low_liquidity_flag")).lower() == "true":
        item["low_liquidity_count"] += 1.0


def _portfolio_timeline_from_aggregate(aggregate: Mapping[int, Mapping[str, float]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for timestamp in sorted(aggregate):
        item = aggregate[timestamp]
        count = max(1.0, _finite(item.get("count")))
        output.append({
            "timestamp_ms": timestamp,
            "timestamp_utc": iso_utc(timestamp),
            "symbols_reporting": int(count),
            "mean_strategy_equity": _finite(item.get("strategy_sum")) / count,
            "mean_baseline_equity": _finite(item.get("baseline_sum")) / count,
            "mean_drawdown": _finite(item.get("drawdown_sum")) / count,
            "low_liquidity_symbol_count": int(_finite(item.get("low_liquidity_count"))),
        })
    return output


def _portfolio_timeline(rows_by_symbol: Sequence[Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    aggregate: dict[int, dict[str, float]] = {}
    for rows in rows_by_symbol:
        for row in rows:
            _update_portfolio_aggregate(aggregate, row)
    return _portfolio_timeline_from_aggregate(aggregate)


def _round_model_candidates(
    objective: ObjectiveSpec,
    strategy: StrategyConfig,
    base_feature_cfg: AdvancedFeatureConfig,
    requested: int,
) -> list[RoundModelCandidate]:
    training = objective.training
    base_epochs = max(1, int(training.epochs if training else 100))
    base_lr = float(training.learning_rate if training else 0.03)
    base_l2 = float(training.l2_penalty if training else 1e-3)
    base_threshold = float(training.signal_threshold if training else strategy.signal_threshold)
    base_label_threshold = max(1e-8, float(base_feature_cfg.label_threshold))
    base_label_lookahead = max(1, int(base_feature_cfg.label_lookahead))
    cost_label_floor = _round_trip_cost_label_floor(strategy)
    min_signal_threshold = {
        "conservative": 0.56,
        "regular": 0.52,
        "aggressive": 0.50,
    }.get(objective.name, 0.52)
    # Keep the first few candidates intentionally diverse. The default smoke
    # budget is small, so the prefix must cover baseline, long-biased,
    # short-biased, and order-flow hypotheses instead of walking one family.
    #
    # Last four fields: minimum label horizon in bars, trailing volatility
    # window in bars, realized-volatility multiplier for dynamic barriers,
    # and focal-loss gamma for rare-event training.
    raw: list[tuple[str, float, float, float, float, float, str, float, float, float, float, int, int, int, int, float, float]] = [
        ("default", 1.0, 1.0, 1.0, 1.0, 1.0, str(base_feature_cfg.label_mode), 0.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0.0, 0.0),
        ("day_trade_frequency_probe_forward", 0.65, 1.10, 1.0, 0.75, 4.0, "forward_return", -0.10, 0.14, 0.10, 0.0, 2, 0, 30, 0, 0.0, 0.0),
        ("day_trade_frequency_probe_downside", 0.65, 1.10, 1.0, 0.75, 4.0, "downside_forward_return", -0.10, 0.14, 0.10, 0.0, 2, 0, 30, 0, 0.0, 0.0),
        ("intraday_activity_triple_barrier", 0.80, 1.15, 1.25, 0.75, 8.0, "triple_barrier", -0.10, 0.12, 0.10, 0.0, 2, 1, 45, 0, 0.0, 0.0),
        ("intraday_downside_triple_barrier", 0.80, 1.15, 1.25, 0.75, 8.0, "downside_triple_barrier", -0.10, 0.12, 0.10, 0.0, 2, 1, 45, 0, 0.0, 0.0),
        ("focal_positive_information_event_barrier", 0.75, 1.00, 2.5, 0.75, 8.0, "event_volatility_triple_barrier", -0.12, 0.16, 0.10, 0.10, 3, 3, 45, 60, 1.75, 2.0),
        ("focal_downside_information_event_barrier", 0.75, 1.00, 2.5, 0.75, 8.0, "downside_event_volatility_triple_barrier", -0.12, 0.16, 0.10, 0.10, 3, 3, 45, 60, 1.75, 2.0),
        ("session_volatility_triple_barrier", 0.85, 1.00, 2.0, 1.00, 12.0, "volatility_triple_barrier", -0.08, 0.18, 0.12, 0.12, 5, 5, 60, 120, 2.5, 0.0),
        ("session_downside_volatility_triple_barrier", 0.85, 1.00, 2.0, 1.00, 12.0, "downside_volatility_triple_barrier", -0.08, 0.18, 0.12, 0.12, 5, 5, 60, 120, 2.5, 0.0),
        ("positive_information_event_barrier", 0.75, 1.00, 2.5, 0.75, 8.0, "event_volatility_triple_barrier", -0.12, 0.16, 0.10, 0.10, 3, 3, 45, 60, 1.75, 0.0),
        ("downside_information_event_barrier", 0.75, 1.00, 2.5, 0.75, 8.0, "downside_event_volatility_triple_barrier", -0.12, 0.16, 0.10, 0.10, 3, 3, 45, 60, 1.75, 0.0),
        ("order_flow_information_event_barrier", 0.90, 1.00, 1.75, 0.90, 12.0, "event_volatility_triple_barrier", -0.10, 0.18, 0.12, 0.12, 5, 5, 75, 90, 2.0, 0.0),
        ("downside_order_flow_information_event_barrier", 0.90, 1.00, 1.75, 0.90, 12.0, "downside_event_volatility_triple_barrier", -0.10, 0.18, 0.12, 0.12, 5, 5, 75, 90, 2.0, 0.0),
        ("frequency_probe_forward", 0.50, 1.10, 1.0, 0.75, 4.0, "forward_return", -0.10, 0.14, 0.10, 0.0, 2, 0, 30, 0, 0.0, 0.0),
        ("intraday_micro_triple_barrier", 0.70, 1.05, 1.5, 0.55, 0.35, "triple_barrier", -0.08, 0.25, 0.16, 0.10, 2, 2, 0, 0, 0.0, 0.0),
        ("intraday_breakout_forward", 0.85, 1.10, 1.0, 0.45, 0.25, "forward_return", -0.10, 0.35, 0.20, 0.15, 1, 1, 0, 0, 0.0, 0.0),
        ("high_conviction_triple_barrier", 1.0, 0.80, 3.0, 1.10, 1.25, "triple_barrier", 0.04, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0.0, 0.0),
        ("lower_lr_more_l2", 0.75, 0.75, 3.0, 1.20, 1.25, str(base_feature_cfg.label_mode), 0.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0.0, 0.0),
        ("short_horizon_forward", 0.50, 1.0, 1.0, 0.75, 0.75, "forward_return", 0.0, 0.75, 0.75, 0.50, 1, 1, 0, 0, 0.0, 0.0),
        ("triple_barrier_base", 1.0, 0.90, 1.5, 1.0, 1.0, "triple_barrier", 0.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0.0, 0.0),
        ("triple_barrier_conservative", 0.75, 0.75, 3.0, 1.25, 1.50, "triple_barrier", 0.0, 1.10, 1.10, 1.0, 0, 0, 0, 0, 0.0, 0.0),
        ("lower_signal_short_forward", 0.65, 1.0, 1.25, 0.70, 0.60, "forward_return", -0.06, 0.65, 0.70, 0.35, 1, 1, 0, 0, 0.0, 0.0),
        ("long_horizon_forward", 1.0, 0.75, 2.0, 1.40, 1.75, "forward_return", 0.0, 1.25, 1.25, 1.0, 0, 0, 0, 0, 0.0, 0.0),
        ("lower_signal_triple_barrier", 0.80, 0.90, 2.0, 0.80, 0.80, "triple_barrier", -0.06, 0.75, 0.75, 0.50, 1, 1, 0, 0, 0.0, 0.0),
    ]
    output: list[RoundModelCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for (
        name,
        epoch_mult,
        lr_mult,
        l2_mult,
        threshold_mult,
        lookahead_mult,
        label_mode,
        signal_offset,
        stop_loss_multiplier,
        take_profit_multiplier,
        cooldown_multiplier,
        min_position_hold_bars,
        flat_signal_exit_grace_bars,
        min_label_lookahead,
        label_volatility_window,
        label_volatility_multiplier,
        focal_gamma,
    ) in raw:
        label_lookahead = max(
            1,
            int(min_label_lookahead),
            int(round(base_label_lookahead * float(lookahead_mult))),
        )
        feature_cfg = replace(
            base_feature_cfg,
            label_threshold=max(1e-8, cost_label_floor, base_label_threshold * float(threshold_mult)),
            label_lookahead=label_lookahead,
            label_mode=str(label_mode),
            label_stop_threshold=(
                max(1e-8, float(base_feature_cfg.label_stop_threshold) * float(threshold_mult))
                if base_feature_cfg.label_stop_threshold is not None
                else None
            ),
            label_volatility_window=max(0, int(label_volatility_window)),
            label_volatility_multiplier=max(0.0, float(label_volatility_multiplier)),
        )
        candidate = RoundModelCandidate(
            name=name,
            feature_cfg=feature_cfg,
            epochs=max(1, int(round(base_epochs * float(epoch_mult)))),
            learning_rate=max(1e-6, base_lr * float(lr_mult)),
            l2_penalty=max(0.0, base_l2 * float(l2_mult)),
            signal_threshold=min(0.95, max(min_signal_threshold, base_threshold + float(signal_offset))),
            stop_loss_multiplier=max(0.05, float(stop_loss_multiplier)),
            take_profit_multiplier=max(0.05, float(take_profit_multiplier)),
            cooldown_multiplier=max(0.0, float(cooldown_multiplier)),
            min_position_hold_bars=max(0, int(min_position_hold_bars)),
            flat_signal_exit_grace_bars=max(0, int(flat_signal_exit_grace_bars)),
            focal_gamma=max(0.0, float(focal_gamma)),
        )
        key = (
            candidate.epochs,
            round(candidate.learning_rate, 12),
            round(candidate.l2_penalty, 12),
            round(candidate.signal_threshold, 12),
            round(candidate.stop_loss_multiplier, 12),
            round(candidate.take_profit_multiplier, 12),
            round(candidate.cooldown_multiplier, 12),
            candidate.min_position_hold_bars,
            candidate.flat_signal_exit_grace_bars,
            advanced_feature_dimension(candidate.feature_cfg),
            candidate.feature_cfg.label_threshold,
            candidate.feature_cfg.label_lookahead,
            candidate.feature_cfg.label_mode,
            candidate.feature_cfg.label_volatility_window,
            round(candidate.feature_cfg.label_volatility_multiplier, 12),
            round(candidate.focal_gamma, 12),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
        if len(output) >= max(1, int(requested)):
            break
    return output


def _round_trip_cost_label_floor(strategy: StrategyConfig, *, multiplier: float = 1.25) -> float:
    """Minimum price move a training label must clear after estimated round-trip friction."""

    def nonnegative_bps(value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return parsed if math.isfinite(parsed) and parsed > 0.0 else 0.0

    taker_fee_bps = nonnegative_bps(getattr(strategy, "taker_fee_bps", 0.0))
    slippage_bps = nonnegative_bps(getattr(strategy, "slippage_bps", 0.0))
    round_trip_bps = 2.0 * (taker_fee_bps + slippage_bps)
    return max(0.0, float(multiplier)) * round_trip_bps / 10_000.0


def _apply_probability_calibration(model: object, calibration: object) -> None:
    if getattr(calibration, "status", "fail") == "fail":
        return
    model.probability_temperature = float(getattr(calibration, "temperature"))
    model.probability_calibration_size = int(getattr(calibration, "rows"))
    model.probability_log_loss_before = float(getattr(calibration, "log_loss_before"))
    model.probability_log_loss_after = float(getattr(calibration, "log_loss_after"))
    model.probability_brier_before = float(getattr(calibration, "brier_before"))
    model.probability_brier_after = float(getattr(calibration, "brier_after"))
    model.probability_ece_before = float(getattr(calibration, "expected_calibration_error_before"))
    model.probability_ece_after = float(getattr(calibration, "expected_calibration_error_after"))
    model.probability_calibration_backend_requested = str(getattr(calibration, "calibration_backend_requested", ""))
    model.probability_calibration_backend_kind = str(getattr(calibration, "calibration_backend_kind", ""))
    model.probability_calibration_backend_device = str(getattr(calibration, "calibration_backend_device", ""))
    model.probability_calibration_backend_reason = str(getattr(calibration, "calibration_backend_reason", ""))


def _candidate_has_downside_positive_label(candidate: RoundModelCandidate) -> bool:
    """Return True when label=1 represents a profitable short-side event."""

    mode = str(candidate.feature_cfg.label_mode or "").strip().lower().replace("-", "_")
    return mode in {
        "downside_forward_return",
        "downside_triple_barrier",
        "downside_volatility_triple_barrier",
        "downside_event_volatility_triple_barrier",
    }


def _orient_candidate_model_for_market_side(model: object, candidate: RoundModelCandidate) -> None:
    """Map candidate label semantics onto the runtime long/high, short/low convention."""

    if not _candidate_has_downside_positive_label(candidate):
        return
    model.probability_inverted = True
    warning = "downside_positive_label_oriented_to_short_side"
    warnings = list(getattr(model, "quality_warnings", []) or [])
    if warning not in warnings:
        warnings.append(warning)
    model.quality_warnings = warnings


def _evaluate_round_model_candidate(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    candidate: RoundModelCandidate,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str,
    batch_size: int,
    require_gpu: bool,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> RoundModelCandidateResult:
    candidate_strategy = _strategy_for_round_candidate(strategy, candidate)
    if status_callback is not None:
        status_callback(
            "feature_generation_started",
            {
                "candle_count": len(candles),
                "candidate_stop_loss_pct": float(candidate_strategy.stop_loss_pct),
                "candidate_take_profit_pct": float(candidate_strategy.take_profit_pct),
                "candidate_cooldown_minutes": int(candidate_strategy.cooldown_minutes),
            },
        )
    rows = make_advanced_rows(
        candles,
        candidate.feature_cfg,
        compute_backend=compute_backend,
        require_accelerated=require_gpu,
    )
    if status_callback is not None:
        status_callback(
            "feature_generation_complete",
            {"row_count": len(rows), "feature_dim": advanced_feature_dimension(candidate.feature_cfg)},
        )
    train_selection_rows, validation_rows = _split_train_validation(rows, validation_fraction=0.25)
    train_rows, selection_rows = _split_train_validation(train_selection_rows, validation_fraction=0.20)
    if not train_rows or not selection_rows or not validation_rows:
        raise ValueError("insufficient rows for train/validation backtest evidence")
    if status_callback is not None:
        status_callback(
            "training_started",
            {
                "train_rows": len(train_rows),
                "selection_rows": len(selection_rows),
                "validation_rows": len(validation_rows),
                "batch_size": int(batch_size),
            },
        )
    model, report = train_advanced(
        train_rows,
        candidate.feature_cfg,
        epochs=candidate.epochs,
        learning_rate=candidate.learning_rate,
        l2_penalty=candidate.l2_penalty,
        validation_rows=selection_rows[: max(1, min(len(selection_rows), 5000))],
        early_stopping_rounds=30,
        compute_backend=compute_backend,
        batch_size=batch_size,
        focal_gamma=candidate.focal_gamma,
    )
    if require_gpu:
        _require_non_cpu_backend(
            getattr(model, "training_backend_kind", ""),
            getattr(model, "training_backend_reason", ""),
            "training",
        )
    if status_callback is not None:
        status_callback(
            "training_complete",
            {
                "training_backend_kind": getattr(model, "training_backend_kind", ""),
                "training_backend_device": getattr(model, "training_backend_device", ""),
                "best_epoch": getattr(model, "best_epoch", None),
            },
        )
        status_callback("threshold_calibration_started", {"selection_rows": len(selection_rows)})
    calibration = calibrate_probability_temperature(
        list(selection_rows[: max(1, min(len(selection_rows), 5000))]),
        model,
        compute_backend=compute_backend,
        batch_size=batch_size,
    )
    if require_gpu and getattr(calibration, "status", "fail") != "fail":
        _require_non_cpu_backend(
            calibration.calibration_backend_kind,
            calibration.calibration_backend_reason,
            "probability_calibration",
        )
    _apply_probability_calibration(model, calibration)
    _orient_candidate_model_for_market_side(model, candidate)
    model.decision_threshold = float(candidate.signal_threshold)
    model.threshold_source = "objective_round_evidence_default"
    threshold_report = calibrate_threshold_for_backtest(
        selection_rows,
        model,
        candidate_strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        baseline_threshold=model.decision_threshold,
        start=0.50 if market_type == "futures" else 0.05,
        end=0.95,
        steps=37,
        min_score_delta=0.0,
        min_closed_trades=objective.min_closed_trades,
        min_trades_per_day=objective.min_trades_per_day,
        compute_backend=compute_backend,
        score_batch_size=batch_size,
    )
    diagnostic_threshold = float(
        getattr(threshold_report, "best_threshold", getattr(threshold_report, "threshold", candidate.signal_threshold))
    )
    diagnostic_score = float(getattr(threshold_report, "best_score", getattr(threshold_report, "score", 0.0)))
    diagnostic_pnl = float(
        getattr(threshold_report, "best_realized_pnl", getattr(threshold_report, "realized_pnl", 0.0))
    )
    diagnostic_trades = int(
        getattr(threshold_report, "best_closed_trades", getattr(threshold_report, "closed_trades", 0)) or 0
    )
    model.threshold_diagnostic_best_threshold = diagnostic_threshold
    model.threshold_diagnostic_best_score = diagnostic_score
    model.threshold_diagnostic_best_pnl = diagnostic_pnl
    model.threshold_diagnostic_best_trades = diagnostic_trades
    model.threshold_diagnostic_best_long_threshold = getattr(threshold_report, "best_long_threshold", None)
    model.threshold_diagnostic_best_short_threshold = getattr(threshold_report, "best_short_threshold", None)
    model.threshold_baseline_score = float(getattr(threshold_report, "baseline_score", 0.0))
    model.threshold_baseline_pnl = float(getattr(threshold_report, "baseline_realized_pnl", 0.0))
    model.threshold_baseline_trades = int(getattr(threshold_report, "baseline_closed_trades", 0) or 0)
    model.threshold_min_closed_trades = int(getattr(threshold_report, "min_closed_trades", 0) or 0)
    model.threshold_min_trades_per_day = float(getattr(threshold_report, "min_trades_per_day", 0.0) or 0.0)
    model.threshold_selected_trades_per_day = float(getattr(threshold_report, "trades_per_day", 0.0) or 0.0)
    model.threshold_best_trades_per_day = float(getattr(threshold_report, "best_trades_per_day", 0.0) or 0.0)
    model.threshold_evaluated_thresholds = int(getattr(threshold_report, "evaluated_thresholds", 0) or 0)
    if status_callback is not None:
        status_callback(
            "threshold_calibration_complete",
            {
                "threshold_accepted": bool(threshold_report.accepted),
                "threshold": float(threshold_report.threshold),
                "closed_trades": int(threshold_report.closed_trades),
                "best_threshold": diagnostic_threshold,
                "best_realized_pnl": diagnostic_pnl,
                "best_closed_trades": diagnostic_trades,
                "probability_calibration_backend_kind": getattr(calibration, "calibration_backend_kind", ""),
                "probability_calibration_backend_device": getattr(calibration, "calibration_backend_device", ""),
                "scoring_backend_kind": threshold_report.scoring_backend_kind,
                "scoring_backend_device": threshold_report.scoring_backend_device,
            },
        )
    if require_gpu:
        _require_non_cpu_backend(
            threshold_report.scoring_backend_kind,
            threshold_report.scoring_backend_reason,
            "threshold_scoring",
        )
    if threshold_report.accepted:
        model.decision_threshold = float(threshold_report.threshold)
        model.long_decision_threshold = getattr(threshold_report, "long_threshold", None)
        model.short_decision_threshold = getattr(threshold_report, "short_threshold", None)
        model.threshold_source = "round_selection_backtest"
        model.threshold_calibration_score = float(threshold_report.score)
        model.threshold_calibration_pnl = float(threshold_report.realized_pnl)
        model.threshold_calibration_trades = int(threshold_report.closed_trades)
    base_result = run_backtest(
        selection_rows,
        model,
        candidate_strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=batch_size,
    )
    if require_gpu:
        _require_non_cpu_backend(
            base_result.scoring_backend_kind,
            base_result.scoring_backend_reason,
            "selection_scoring",
        )
    if status_callback is not None:
        status_callback(
            "selection_backtest_complete",
            {
                "base_pnl": float(base_result.realized_pnl),
                "base_drawdown": float(base_result.max_drawdown),
                "scoring_backend_kind": base_result.scoring_backend_kind,
                "scoring_backend_device": base_result.scoring_backend_device,
            },
        )
    base_reject_reason = objective.reject_reason(base_result)
    base_accepts = base_reject_reason is None
    base_raw_score = objective.score(base_result)
    base_score = base_raw_score if base_accepts else float("-inf")
    inverted_model = model
    inverted_result = base_result
    inverted_reject_reason = "downside_label_orientation_locked"
    inverted_raw_score = float("-inf")
    inverted_score = float("-inf")
    if _candidate_has_downside_positive_label(candidate):
        if status_callback is not None:
            status_callback(
                "inversion_backtest_skipped",
                {"reason": "downside_label_orientation_locked"},
            )
    else:
        inverted_model = copy.deepcopy(model)
        inverted_model.probability_inverted = not bool(getattr(inverted_model, "probability_inverted", False))
        inverted_model.model_family = f"{inverted_model.model_family}:round_selection_inverted"
        inverted_model.quality_warnings = [
            *list(getattr(inverted_model, "quality_warnings", [])),
            "round_selection_probability_inversion_variant",
        ]
        inverted_result = run_backtest(
            selection_rows,
            inverted_model,
            candidate_strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=batch_size,
        )
        if require_gpu:
            _require_non_cpu_backend(
                inverted_result.scoring_backend_kind,
                inverted_result.scoring_backend_reason,
                "inversion_scoring",
            )
        if status_callback is not None:
            status_callback(
                "inversion_backtest_complete",
                {
                    "inverted_pnl": float(inverted_result.realized_pnl),
                    "inverted_drawdown": float(inverted_result.max_drawdown),
                    "scoring_backend_kind": inverted_result.scoring_backend_kind,
                    "scoring_backend_device": inverted_result.scoring_backend_device,
                },
            )
        inverted_reject_reason = objective.reject_reason(inverted_result)
        inverted_accepts = inverted_reject_reason is None
        inverted_raw_score = objective.score(inverted_result)
        inverted_score = inverted_raw_score if inverted_accepts else float("-inf")
    score = base_score
    selection_result = base_result
    selection_reject_reason = base_reject_reason
    if inverted_score > base_score + 1e-12:
        model = inverted_model
        score = inverted_score
        selection_result = inverted_result
        selection_reject_reason = inverted_reject_reason
        model.round_selection_gate_passed = True
        model.round_selection_reject_reason = ""
    elif base_accepts:
        selection_result = base_result
        selection_reject_reason = base_reject_reason
        model.round_selection_gate_passed = True
        model.round_selection_reject_reason = ""
    elif not math.isfinite(base_score):
        if inverted_raw_score > base_raw_score + 1e-12:
            model = inverted_model
            score = inverted_raw_score
            chosen_reject_reason = inverted_reject_reason
            selection_result = inverted_result
            selection_reject_reason = inverted_reject_reason
        else:
            score = base_raw_score
            chosen_reject_reason = base_reject_reason
            selection_result = base_result
            selection_reject_reason = base_reject_reason
        model.round_selection_gate_passed = False
        model.round_selection_reject_reason = str(chosen_reject_reason or "selection_gate_failed")
        if diagnostic_trades > 0:
            model.threshold_source = "round_selection_rejected_no_entry_diagnostic_recorded"
            model.threshold_calibration_score = diagnostic_score
            model.threshold_calibration_pnl = diagnostic_pnl
            model.threshold_calibration_trades = diagnostic_trades
        else:
            model.threshold_source = "round_selection_rejected_no_entry"
        model.decision_threshold = 1.0
        model.long_decision_threshold = 1.0
        model.short_decision_threshold = None
        model.quality_warnings = [
            *list(getattr(model, "quality_warnings", [])),
            "round_selection_gate_failed_diagnostic_holdout_only",
            "round_selection_failed_no_entry_enforced",
        ]
    model.strategy_overrides = strategy_overrides_from_config(candidate_strategy)
    return RoundModelCandidateResult(
        candidate=candidate,
        score=float(score),
        model=model,
        report=report,
        selection_result=selection_result,
        selection_reject_reason=selection_reject_reason,
        training_rows=list(train_rows),
        selection_rows=list(selection_rows),
        rows=list(rows),
        validation_rows=list(validation_rows),
    )


def _round_candidate_rank_key(result: RoundModelCandidateResult) -> tuple[float, ...]:
    """Rank candidates without allowing failed risk gates to masquerade as live evidence."""

    model = result.model
    selection = result.selection_result

    def finite(value: object, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return parsed if math.isfinite(parsed) else default

    gate_passed = 1.0 if bool(getattr(model, "round_selection_gate_passed", False)) else 0.0
    score = finite(result.score, float("-inf"))
    diagnostic_pnl = finite(getattr(model, "threshold_diagnostic_best_pnl", None), float("-inf"))
    diagnostic_trades = finite(getattr(model, "threshold_diagnostic_best_trades", 0), 0.0)
    selection_pnl = finite(getattr(selection, "realized_pnl", 0.0), 0.0)
    closed_trades = finite(getattr(selection, "closed_trades", 0), 0.0)
    liquidation_events = finite(getattr(selection, "liquidation_events", 0), 0.0)
    drawdown = finite(getattr(selection, "max_drawdown", 1.0), 1.0)
    fees = finite(getattr(selection, "total_fees", 0.0), 0.0)
    if gate_passed <= 0.0:
        profitable_selection = 1.0 if selection_pnl > 0.0 else 0.0
        nonnegative_selection = 1.0 if selection_pnl >= 0.0 and liquidation_events <= 0.0 else 0.0
        return (
            gate_passed,
            profitable_selection,
            nonnegative_selection,
            selection_pnl,
            -liquidation_events,
            -drawdown,
            -fees,
            diagnostic_pnl,
            score,
            diagnostic_trades,
            -closed_trades,
        )
    return (
        gate_passed,
        score,
        diagnostic_pnl,
        selection_pnl,
        diagnostic_trades,
        closed_trades,
        -liquidation_events,
        -drawdown,
        -fees,
    )


def _append_unique_warning(model: object, warning: str) -> None:
    warnings = list(getattr(model, "quality_warnings", []) or [])
    if warning not in warnings:
        warnings.append(warning)
    model.quality_warnings = warnings


def _finite_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _hybrid_search_seed_model(result: RoundModelCandidateResult, *, market_type: str) -> object:
    """Return a model configured for offline hybrid rescue search.

    Failed base candidates are forced to a no-entry threshold before they can
    ever be serialized for live use. That is correct for safety, but it makes
    offline hybrid research impossible unless we search from the best diagnostic
    threshold captured during selection. The returned copy is used only for
    selection backtests; the original fail-closed model remains unchanged unless
    the hybrid result passes the full objective gates.
    """

    model = copy.deepcopy(result.model)
    if bool(getattr(result.model, "round_selection_gate_passed", True)):
        return model
    diagnostic_threshold = _finite(
        getattr(model, "threshold_diagnostic_best_threshold", None),
        float(result.candidate.signal_threshold),
    )
    if str(market_type).lower() == "futures":
        best_long = getattr(model, "threshold_diagnostic_best_long_threshold", None)
        best_short = getattr(model, "threshold_diagnostic_best_short_threshold", None)
        if best_long is None and best_short is None:
            threshold = max(0.5, min(0.95, float(diagnostic_threshold)))
            model.long_decision_threshold = threshold
            model.short_decision_threshold = 1.0 - threshold
        else:
            model.long_decision_threshold = best_long
            model.short_decision_threshold = best_short
    else:
        model.long_decision_threshold = None
        model.short_decision_threshold = None
    model.decision_threshold = max(0.0, min(1.0, float(diagnostic_threshold)))
    model.round_selection_gate_passed = True
    model.round_selection_reject_reason = ""
    model.threshold_source = "hybrid_rescue_diagnostic_threshold_search"
    _append_unique_warning(model, "hybrid_rescue_search_from_rejected_base")
    return model


def _select_hybrid_model_zoo_if_accepted(
    result: RoundModelCandidateResult,
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str,
    batch_size: int,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> RoundModelCandidateResult:
    """Attach the adaptive hybrid expert pack when it passes full objective gates."""

    model = result.model
    feature_dim = int(getattr(model, "feature_dim", 0) or 0)
    base_selection_failed = bool(getattr(model, "round_selection_gate_passed", True)) is False
    if feature_dim <= 1 or not result.training_rows or not result.selection_rows:
        if status_callback is not None:
            status_callback(
                "hybrid_model_zoo_skipped",
                {
                    "reason": "insufficient_feature_or_split_evidence",
                    "feature_dim": int(feature_dim),
                    "training_rows": len(result.training_rows),
                    "selection_rows": len(result.selection_rows),
                },
            )
        return result
    if status_callback is not None:
        status_callback(
            "hybrid_model_zoo_started",
            {
                "feature_dim": int(feature_dim),
                "training_rows": len(result.training_rows),
                "selection_rows": len(result.selection_rows),
                "base_selection_failed": bool(base_selection_failed),
            },
        )
    search_model = _hybrid_search_seed_model(result, market_type=market_type)
    hybrid_report = optimize_hybrid_model_zoo(
        search_model,
        result.training_rows,
        result.selection_rows,
        strategy,
        objective_name=objective.name,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=batch_size,
        feature_count=min(13, feature_dim),
    )
    model.hybrid_profile = str(hybrid_report.best_profile)
    model.hybrid_base_score = _finite_or_none(hybrid_report.base_score)
    model.hybrid_best_score = _finite_or_none(hybrid_report.best_score)
    model.hybrid_evaluated_profiles = int(hybrid_report.evaluated_profiles)
    selected_result = getattr(hybrid_report, "best_result", None)
    if not bool(getattr(hybrid_report, "accepted", False)) or selected_result is None:
        _append_unique_warning(model, "round_hybrid_model_zoo_rejected")
        if status_callback is not None:
            status_callback(
                "hybrid_model_zoo_complete",
                {
                    "accepted": False,
                    "profile": str(hybrid_report.best_profile),
                    "base_score": model.hybrid_base_score,
                    "best_score": model.hybrid_best_score,
                    "evaluated_profiles": int(hybrid_report.evaluated_profiles),
                },
            )
        return replace(result, model=model)
    if status_callback is not None:
        status_callback(
            "hybrid_model_zoo_complete",
            {
                "accepted": True,
                "profile": str(hybrid_report.best_profile),
                "base_score": _finite_or_none(hybrid_report.base_score),
                "best_score": _finite_or_none(hybrid_report.best_score),
                "evaluated_profiles": int(hybrid_report.evaluated_profiles),
            },
        )
    hybrid_model = hybrid_report.model
    hybrid_model.hybrid_profile = str(hybrid_report.best_profile)
    hybrid_model.hybrid_base_score = _finite_or_none(hybrid_report.base_score)
    hybrid_model.hybrid_best_score = _finite_or_none(hybrid_report.best_score)
    hybrid_model.hybrid_evaluated_profiles = int(hybrid_report.evaluated_profiles)
    hybrid_model.round_selection_gate_passed = True
    hybrid_model.round_selection_reject_reason = ""
    _append_unique_warning(hybrid_model, "round_hybrid_model_zoo_selected")
    if base_selection_failed:
        _append_unique_warning(hybrid_model, "round_hybrid_model_zoo_rescued_rejected_base")
    return replace(
        result,
        score=float(hybrid_report.best_score),
        model=hybrid_model,
        selection_result=selected_result,
        selection_reject_reason=None,
    )


def _select_rule_alpha_model_zoo_if_accepted(
    result: RoundModelCandidateResult,
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str,
    batch_size: int,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> RoundModelCandidateResult:
    """Promote an interpretable alpha template only if it passes selection gates."""

    if not result.selection_rows:
        return result
    if status_callback is not None:
        status_callback(
            "rule_alpha_model_zoo_started",
            {"selection_rows": len(result.selection_rows), "max_candidates": DEFAULT_RULE_ALPHA_MAX_CANDIDATES},
        )
    alpha_report = optimize_rule_alpha_model_zoo(
        result.selection_rows,
        strategy,
        objective_name=objective.name,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=batch_size,
        max_candidates=DEFAULT_RULE_ALPHA_MAX_CANDIDATES,
        feature_cfg=result.candidate.feature_cfg,
    )
    model = result.model
    model.rule_alpha_evaluated_candidates = int(alpha_report.evaluated_candidates)
    model.rule_alpha_candidate_summary = summarize_rule_alpha_candidate_distribution(alpha_report.candidate_results)
    if alpha_report.best_candidate is not None:
        model.rule_alpha_profile = str(alpha_report.best_candidate.name)
        model.rule_alpha_family = str(alpha_report.best_candidate.family)
        model.rule_alpha_best_score = float(alpha_report.best_score)
        model.rule_alpha_probability_inverted = bool(alpha_report.best_probability_inverted)
        model.rule_alpha_best_reject_reason = str(alpha_report.best_reject_reason or "")
    if alpha_report.best_result is not None:
        model.rule_alpha_best_pnl = float(alpha_report.best_result.realized_pnl)
        model.rule_alpha_best_closed_trades = int(alpha_report.best_result.closed_trades)
        path_summary = summarize_rule_alpha_trade_path(alpha_report.best_result)
        model.rule_alpha_best_win_rate = float(path_summary["win_rate"])
        model.rule_alpha_best_profit_factor = float(path_summary["profit_factor"])
        model.rule_alpha_best_max_drawdown = float(path_summary["max_drawdown"])
        model.rule_alpha_best_exit_reason_counts = dict(path_summary["exit_reason_counts"])
        model.rule_alpha_best_side_counts = dict(path_summary["side_counts"])
    if not alpha_report.accepted or alpha_report.model is None or alpha_report.best_result is None:
        _append_unique_warning(model, "rule_alpha_model_zoo_rejected")
        if status_callback is not None:
            status_callback(
                "rule_alpha_model_zoo_complete",
                {
                    "accepted": False,
                    "evaluated_candidates": int(alpha_report.evaluated_candidates),
                    "best_score": float(alpha_report.best_score),
                    "best_candidate": (
                        alpha_report.best_candidate.name
                        if alpha_report.best_candidate is not None
                        else ""
                    ),
                },
            )
        return result
    if math.isfinite(result.score) and alpha_report.best_score <= result.score + 1e-12:
        _append_unique_warning(model, "rule_alpha_model_zoo_not_better_than_selected_model")
        if status_callback is not None:
            status_callback(
                "rule_alpha_model_zoo_complete",
                {
                    "accepted": True,
                    "promoted": False,
                    "evaluated_candidates": int(alpha_report.evaluated_candidates),
                    "best_score": float(alpha_report.best_score),
                    "selected_score": float(result.score),
                    "best_candidate": alpha_report.best_candidate.name if alpha_report.best_candidate else "",
                },
            )
        return result
    alpha_model = alpha_report.model
    alpha_model.model_candidate_count = int(getattr(model, "model_candidate_count", 1) or 1)
    alpha_model.model_selected_candidate = f"{result.candidate.name}+rule_alpha:{alpha_model.rule_alpha_profile}"
    alpha_model.model_selection_score = float(alpha_report.best_score)
    alpha_model.round_candidate_diagnostics = list(getattr(model, "round_candidate_diagnostics", []) or [])
    alpha_model.quality_warnings = [
        *list(getattr(alpha_model, "quality_warnings", []) or []),
        "rule_alpha_model_zoo_selected",
    ]
    if status_callback is not None:
        status_callback(
            "rule_alpha_model_zoo_complete",
            {
                "accepted": True,
                "promoted": True,
                "evaluated_candidates": int(alpha_report.evaluated_candidates),
                "best_score": float(alpha_report.best_score),
                "best_candidate": alpha_report.best_candidate.name if alpha_report.best_candidate else "",
                "realized_pnl": float(alpha_report.best_result.realized_pnl),
                "closed_trades": int(alpha_report.best_result.closed_trades),
            },
        )
    return replace(
        result,
        score=float(alpha_report.best_score),
        model=alpha_model,
        selection_result=alpha_report.best_result,
        selection_reject_reason=None,
    )


def _round_candidate_diagnostic(
    result: RoundModelCandidateResult,
    *,
    strategy: StrategyConfig,
    selected: bool,
) -> dict[str, object]:
    candidate_strategy = _strategy_for_round_candidate(strategy, result.candidate)
    model = result.model
    selection = result.selection_result
    rule_alpha_summary = (
        dict(getattr(model, "rule_alpha_candidate_summary", {}) or {})
        if isinstance(getattr(model, "rule_alpha_candidate_summary", {}), dict)
        else {}
    )
    return {
        "name": result.candidate.name,
        "selected": bool(selected),
        "score": float(result.score),
        "signal_threshold": float(result.candidate.signal_threshold),
        "stop_loss_pct": float(candidate_strategy.stop_loss_pct),
        "take_profit_pct": float(candidate_strategy.take_profit_pct),
        "cooldown_minutes": int(candidate_strategy.cooldown_minutes),
        "min_position_hold_bars": int(candidate_strategy.min_position_hold_bars),
        "flat_signal_exit_grace_bars": int(candidate_strategy.flat_signal_exit_grace_bars),
        "label_threshold": float(result.candidate.feature_cfg.label_threshold),
        "label_lookahead": int(result.candidate.feature_cfg.label_lookahead),
        "label_mode": str(result.candidate.feature_cfg.label_mode),
        "focal_gamma": float(result.candidate.focal_gamma),
        "model_family": str(getattr(model, "model_family", "") or ""),
        "probability_inverted": bool(getattr(model, "probability_inverted", False)),
        "hybrid_profile": str(getattr(model, "hybrid_profile", "") or ""),
        "hybrid_base_score": (
            float(getattr(model, "hybrid_base_score"))
            if getattr(model, "hybrid_base_score", None) is not None
            else None
        ),
        "hybrid_best_score": (
            float(getattr(model, "hybrid_best_score"))
            if getattr(model, "hybrid_best_score", None) is not None
            else None
        ),
        "hybrid_evaluated_profiles": int(getattr(model, "hybrid_evaluated_profiles", 0) or 0),
        "hybrid_expert_count": len(getattr(model, "hybrid_experts", []) or []),
        "rule_alpha_profile": str(getattr(model, "rule_alpha_profile", "") or ""),
        "rule_alpha_family": str(getattr(model, "rule_alpha_family", "") or ""),
        "rule_alpha_best_score": (
            float(getattr(model, "rule_alpha_best_score"))
            if getattr(model, "rule_alpha_best_score", None) is not None
            else None
        ),
        "rule_alpha_best_pnl": (
            float(getattr(model, "rule_alpha_best_pnl"))
            if getattr(model, "rule_alpha_best_pnl", None) is not None
            else None
        ),
        "rule_alpha_best_closed_trades": int(getattr(model, "rule_alpha_best_closed_trades", 0) or 0),
        "rule_alpha_best_win_rate": (
            float(getattr(model, "rule_alpha_best_win_rate"))
            if getattr(model, "rule_alpha_best_win_rate", None) is not None
            else None
        ),
        "rule_alpha_best_profit_factor": (
            float(getattr(model, "rule_alpha_best_profit_factor"))
            if getattr(model, "rule_alpha_best_profit_factor", None) is not None
            else None
        ),
        "rule_alpha_best_max_drawdown": (
            float(getattr(model, "rule_alpha_best_max_drawdown"))
            if getattr(model, "rule_alpha_best_max_drawdown", None) is not None
            else None
        ),
        "rule_alpha_best_exit_reason_counts": dict(getattr(model, "rule_alpha_best_exit_reason_counts", {}) or {}),
        "rule_alpha_best_side_counts": dict(getattr(model, "rule_alpha_best_side_counts", {}) or {}),
        "rule_alpha_best_reject_reason": str(getattr(model, "rule_alpha_best_reject_reason", "") or ""),
        "rule_alpha_probability_inverted": bool(getattr(model, "rule_alpha_probability_inverted", False)),
        "rule_alpha_evaluated_candidates": int(getattr(model, "rule_alpha_evaluated_candidates", 0) or 0),
        "rule_alpha_active_candidates": int(_finite(rule_alpha_summary.get("active_candidates"))),
        "rule_alpha_profitable_candidates": int(_finite(rule_alpha_summary.get("profitable_candidates"))),
        "rule_alpha_accepted_candidates": int(_finite(rule_alpha_summary.get("accepted_candidates"))),
        "rule_alpha_event_candidates_with_signals": int(_finite(rule_alpha_summary.get("event_candidates_with_signals"))),
        "rule_alpha_event_positive_candidates": int(_finite(rule_alpha_summary.get("event_positive_candidates"))),
        "rule_alpha_event_best_candidate": str(rule_alpha_summary.get("event_best_candidate", "") or ""),
        "rule_alpha_event_best_net_edge_bps": _finite(rule_alpha_summary.get("event_best_net_edge_bps")),
        "rule_alpha_event_best_signal_count": int(_finite(rule_alpha_summary.get("event_best_signal_count"))),
        "rule_alpha_event_best_hit_rate": _finite(rule_alpha_summary.get("event_best_hit_rate")),
        "rule_alpha_event_best_horizon_bars": int(_finite(rule_alpha_summary.get("event_best_horizon_bars"))),
        "rule_alpha_event_best_probability_inverted": bool(rule_alpha_summary.get("event_best_probability_inverted") is True),
        "rule_alpha_max_closed_trades": int(_finite(rule_alpha_summary.get("max_closed_trades"))),
        "rule_alpha_most_active_candidate": str(rule_alpha_summary.get("most_active_candidate", "") or ""),
        "rule_alpha_most_active_pnl": _finite(rule_alpha_summary.get("most_active_pnl")),
        "rule_alpha_most_active_profit_factor": _finite(rule_alpha_summary.get("most_active_profit_factor")),
        "rule_alpha_most_active_reject_reason": str(rule_alpha_summary.get("most_active_reject_reason", "") or ""),
        "rule_alpha_best_pnl_candidate": str(rule_alpha_summary.get("best_pnl_candidate", "") or ""),
        "rule_alpha_best_pnl_candidate_pnl": _finite(rule_alpha_summary.get("best_pnl")),
        "rule_alpha_best_pnl_candidate_closed_trades": int(_finite(rule_alpha_summary.get("best_pnl_closed_trades"))),
        "rule_alpha_families_with_trades": str(rule_alpha_summary.get("families_with_trades", "") or ""),
        "rule_alpha_profiles_with_trades": str(rule_alpha_summary.get("profiles_with_trades", "") or ""),
        "round_selection_gate_passed": bool(getattr(model, "round_selection_gate_passed", False)),
        "round_selection_reject_reason": str(getattr(model, "round_selection_reject_reason", "") or ""),
        "threshold_source": str(getattr(model, "threshold_source", "") or ""),
        "decision_threshold": (
            float(getattr(model, "decision_threshold"))
            if getattr(model, "decision_threshold", None) is not None
            else None
        ),
        "long_decision_threshold": (
            float(getattr(model, "long_decision_threshold"))
            if getattr(model, "long_decision_threshold", None) is not None
            else None
        ),
        "short_decision_threshold": (
            float(getattr(model, "short_decision_threshold"))
            if getattr(model, "short_decision_threshold", None) is not None
            else None
        ),
        "threshold_diagnostic_best_long_threshold": (
            float(getattr(model, "threshold_diagnostic_best_long_threshold"))
            if getattr(model, "threshold_diagnostic_best_long_threshold", None) is not None
            else None
        ),
        "threshold_diagnostic_best_short_threshold": (
            float(getattr(model, "threshold_diagnostic_best_short_threshold"))
            if getattr(model, "threshold_diagnostic_best_short_threshold", None) is not None
            else None
        ),
        "threshold_calibration_score": (
            float(getattr(model, "threshold_calibration_score"))
            if getattr(model, "threshold_calibration_score", None) is not None
            else None
        ),
        "threshold_calibration_pnl": (
            float(getattr(model, "threshold_calibration_pnl"))
            if getattr(model, "threshold_calibration_pnl", None) is not None
            else None
        ),
        "threshold_calibration_trades": int(getattr(model, "threshold_calibration_trades", 0) or 0),
        "threshold_baseline_score": (
            float(getattr(model, "threshold_baseline_score"))
            if getattr(model, "threshold_baseline_score", None) is not None
            else None
        ),
        "threshold_baseline_pnl": (
            float(getattr(model, "threshold_baseline_pnl"))
            if getattr(model, "threshold_baseline_pnl", None) is not None
            else None
        ),
        "threshold_baseline_trades": int(getattr(model, "threshold_baseline_trades", 0) or 0),
        "threshold_min_closed_trades": int(getattr(model, "threshold_min_closed_trades", 0) or 0),
        "threshold_min_trades_per_day": float(getattr(model, "threshold_min_trades_per_day", 0.0) or 0.0),
        "threshold_selected_trades_per_day": float(getattr(model, "threshold_selected_trades_per_day", 0.0) or 0.0),
        "threshold_best_trades_per_day": float(getattr(model, "threshold_best_trades_per_day", 0.0) or 0.0),
        "threshold_evaluated_thresholds": int(getattr(model, "threshold_evaluated_thresholds", 0) or 0),
        "threshold_diagnostic_best_threshold": (
            float(getattr(model, "threshold_diagnostic_best_threshold"))
            if getattr(model, "threshold_diagnostic_best_threshold", None) is not None
            else None
        ),
        "threshold_diagnostic_best_score": (
            float(getattr(model, "threshold_diagnostic_best_score"))
            if getattr(model, "threshold_diagnostic_best_score", None) is not None
            else None
        ),
        "threshold_diagnostic_best_pnl": (
            float(getattr(model, "threshold_diagnostic_best_pnl"))
            if getattr(model, "threshold_diagnostic_best_pnl", None) is not None
            else None
        ),
        "threshold_diagnostic_best_trades": int(getattr(model, "threshold_diagnostic_best_trades", 0) or 0),
        "selection_realized_pnl": float(getattr(selection, "realized_pnl", 0.0)),
        "selection_closed_trades": int(getattr(selection, "closed_trades", 0) or 0),
        "selection_max_drawdown": float(getattr(selection, "max_drawdown", 0.0)),
        "selection_win_rate": float(getattr(selection, "win_rate", 0.0)),
        "selection_total_fees": float(getattr(selection, "total_fees", 0.0)),
        "selection_profit_factor": float(getattr(selection, "profit_factor", 0.0)),
        "selection_expectancy": float(getattr(selection, "expectancy", 0.0)),
        "selection_edge_vs_buy_hold": float(getattr(selection, "edge_vs_buy_hold", 0.0)),
        "selection_trades_per_day": float(closed_trades_per_day(selection)),
        "selection_trades_per_day_cap_hit": int(getattr(selection, "trades_per_day_cap_hit", 0) or 0),
        "selection_regime_entry_skips": int(getattr(selection, "regime_entry_skips", 0) or 0),
        "selection_meta_label_skips": int(getattr(selection, "meta_label_skips", 0) or 0),
        "selection_meta_label_downsizes": int(getattr(selection, "meta_label_downsizes", 0) or 0),
        "selection_liquidation_events": int(getattr(selection, "liquidation_events", 0) or 0),
        "selection_reject_reason": str(result.selection_reject_reason or ""),
    }


def _candidate_diagnostic_rows(symbol: str, model: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    diagnostics = getattr(model, "round_candidate_diagnostics", []) or []
    if not isinstance(diagnostics, list):
        return rows
    for index, item in enumerate(diagnostics, start=1):
        if not isinstance(item, Mapping):
            continue
        rows.append({
            "symbol": str(symbol),
            "candidate_index": int(index),
            "name": str(item.get("name", "")),
            "selected": bool(item.get("selected") is True),
            "score": _finite(item.get("score")),
            "signal_threshold": _finite(item.get("signal_threshold")),
            "stop_loss_pct": _finite(item.get("stop_loss_pct")),
            "take_profit_pct": _finite(item.get("take_profit_pct")),
            "cooldown_minutes": int(_finite(item.get("cooldown_minutes"))),
            "min_position_hold_bars": int(_finite(item.get("min_position_hold_bars"))),
            "flat_signal_exit_grace_bars": int(_finite(item.get("flat_signal_exit_grace_bars"))),
            "label_threshold": _finite(item.get("label_threshold")),
            "label_lookahead": int(_finite(item.get("label_lookahead"))),
            "label_mode": str(item.get("label_mode", "")),
            "focal_gamma": _finite(item.get("focal_gamma")),
            "model_family": str(item.get("model_family", "")),
            "probability_inverted": bool(item.get("probability_inverted") is True),
            "hybrid_profile": str(item.get("hybrid_profile", "")),
            "hybrid_base_score": item.get("hybrid_base_score"),
            "hybrid_best_score": item.get("hybrid_best_score"),
            "hybrid_evaluated_profiles": int(_finite(item.get("hybrid_evaluated_profiles"))),
            "hybrid_expert_count": int(_finite(item.get("hybrid_expert_count"))),
            "rule_alpha_profile": str(item.get("rule_alpha_profile", "")),
            "rule_alpha_family": str(item.get("rule_alpha_family", "")),
            "rule_alpha_best_score": item.get("rule_alpha_best_score"),
            "rule_alpha_best_pnl": item.get("rule_alpha_best_pnl"),
            "rule_alpha_best_closed_trades": int(_finite(item.get("rule_alpha_best_closed_trades"))),
            "rule_alpha_best_win_rate": item.get("rule_alpha_best_win_rate"),
            "rule_alpha_best_profit_factor": item.get("rule_alpha_best_profit_factor"),
            "rule_alpha_best_max_drawdown": item.get("rule_alpha_best_max_drawdown"),
            "rule_alpha_best_exit_reason_counts": json.dumps(
                item.get("rule_alpha_best_exit_reason_counts", {}) or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "rule_alpha_best_side_counts": json.dumps(
                item.get("rule_alpha_best_side_counts", {}) or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "rule_alpha_best_reject_reason": str(item.get("rule_alpha_best_reject_reason", "")),
            "rule_alpha_probability_inverted": bool(item.get("rule_alpha_probability_inverted") is True),
            "rule_alpha_evaluated_candidates": int(_finite(item.get("rule_alpha_evaluated_candidates"))),
            "rule_alpha_active_candidates": int(_finite(item.get("rule_alpha_active_candidates"))),
            "rule_alpha_profitable_candidates": int(_finite(item.get("rule_alpha_profitable_candidates"))),
            "rule_alpha_accepted_candidates": int(_finite(item.get("rule_alpha_accepted_candidates"))),
            "rule_alpha_event_candidates_with_signals": int(_finite(item.get("rule_alpha_event_candidates_with_signals"))),
            "rule_alpha_event_positive_candidates": int(_finite(item.get("rule_alpha_event_positive_candidates"))),
            "rule_alpha_event_best_candidate": str(item.get("rule_alpha_event_best_candidate", "")),
            "rule_alpha_event_best_net_edge_bps": _finite(item.get("rule_alpha_event_best_net_edge_bps")),
            "rule_alpha_event_best_signal_count": int(_finite(item.get("rule_alpha_event_best_signal_count"))),
            "rule_alpha_event_best_hit_rate": _finite(item.get("rule_alpha_event_best_hit_rate")),
            "rule_alpha_event_best_horizon_bars": int(_finite(item.get("rule_alpha_event_best_horizon_bars"))),
            "rule_alpha_event_best_probability_inverted": bool(item.get("rule_alpha_event_best_probability_inverted") is True),
            "rule_alpha_max_closed_trades": int(_finite(item.get("rule_alpha_max_closed_trades"))),
            "rule_alpha_most_active_candidate": str(item.get("rule_alpha_most_active_candidate", "")),
            "rule_alpha_most_active_pnl": _finite(item.get("rule_alpha_most_active_pnl")),
            "rule_alpha_most_active_profit_factor": _finite(item.get("rule_alpha_most_active_profit_factor")),
            "rule_alpha_most_active_reject_reason": str(item.get("rule_alpha_most_active_reject_reason", "")),
            "rule_alpha_best_pnl_candidate": str(item.get("rule_alpha_best_pnl_candidate", "")),
            "rule_alpha_best_pnl_candidate_pnl": _finite(item.get("rule_alpha_best_pnl_candidate_pnl")),
            "rule_alpha_best_pnl_candidate_closed_trades": int(_finite(item.get("rule_alpha_best_pnl_candidate_closed_trades"))),
            "rule_alpha_families_with_trades": str(item.get("rule_alpha_families_with_trades", "")),
            "rule_alpha_profiles_with_trades": str(item.get("rule_alpha_profiles_with_trades", "")),
            "round_selection_gate_passed": bool(item.get("round_selection_gate_passed") is True),
            "round_selection_reject_reason": str(item.get("round_selection_reject_reason", "")),
            "threshold_source": str(item.get("threshold_source", "")),
            "decision_threshold": item.get("decision_threshold"),
            "long_decision_threshold": item.get("long_decision_threshold"),
            "short_decision_threshold": item.get("short_decision_threshold"),
            "threshold_calibration_score": item.get("threshold_calibration_score"),
            "threshold_calibration_pnl": item.get("threshold_calibration_pnl"),
            "threshold_calibration_trades": int(_finite(item.get("threshold_calibration_trades"))),
            "threshold_baseline_score": item.get("threshold_baseline_score"),
            "threshold_baseline_pnl": item.get("threshold_baseline_pnl"),
            "threshold_baseline_trades": int(_finite(item.get("threshold_baseline_trades"))),
            "threshold_min_closed_trades": int(_finite(item.get("threshold_min_closed_trades"))),
            "threshold_min_trades_per_day": _finite(item.get("threshold_min_trades_per_day")),
            "threshold_selected_trades_per_day": _finite(item.get("threshold_selected_trades_per_day")),
            "threshold_best_trades_per_day": _finite(item.get("threshold_best_trades_per_day")),
            "threshold_evaluated_thresholds": int(_finite(item.get("threshold_evaluated_thresholds"))),
            "threshold_diagnostic_best_threshold": item.get("threshold_diagnostic_best_threshold"),
            "threshold_diagnostic_best_long_threshold": item.get("threshold_diagnostic_best_long_threshold"),
            "threshold_diagnostic_best_short_threshold": item.get("threshold_diagnostic_best_short_threshold"),
            "threshold_diagnostic_best_score": item.get("threshold_diagnostic_best_score"),
            "threshold_diagnostic_best_pnl": item.get("threshold_diagnostic_best_pnl"),
            "threshold_diagnostic_best_trades": int(_finite(item.get("threshold_diagnostic_best_trades"))),
            "selection_realized_pnl": _finite(item.get("selection_realized_pnl")),
            "selection_closed_trades": int(_finite(item.get("selection_closed_trades"))),
            "selection_max_drawdown": _finite(item.get("selection_max_drawdown")),
            "selection_win_rate": _finite(item.get("selection_win_rate")),
            "selection_total_fees": _finite(item.get("selection_total_fees")),
            "selection_profit_factor": _finite(item.get("selection_profit_factor")),
            "selection_expectancy": _finite(item.get("selection_expectancy")),
            "selection_edge_vs_buy_hold": _finite(item.get("selection_edge_vs_buy_hold")),
            "selection_trades_per_day": _finite(item.get("selection_trades_per_day")),
            "selection_trades_per_day_cap_hit": int(_finite(item.get("selection_trades_per_day_cap_hit"))),
            "selection_regime_entry_skips": int(_finite(item.get("selection_regime_entry_skips"))),
            "selection_meta_label_skips": int(_finite(item.get("selection_meta_label_skips"))),
            "selection_meta_label_downsizes": int(_finite(item.get("selection_meta_label_downsizes"))),
            "selection_liquidation_events": int(_finite(item.get("selection_liquidation_events"))),
            "selection_reject_reason": str(item.get("selection_reject_reason", "")),
        })
    return rows


def train_round_model(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str,
    batch_size: int,
    require_gpu: bool = False,
    model_candidate_count: int = 1,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[object, object, list[object], list[object]]:
    base_feature_cfg = default_config_for(objective.name, strategy.enabled_features)
    candidates = _round_model_candidates(objective, strategy, base_feature_cfg, model_candidate_count)
    if len(candidates) > 1 and status_callback is not None:
        status_callback("model_candidate_search_started", {"candidate_count": len(candidates)})
    best: RoundModelCandidateResult | None = None
    evaluated: list[RoundModelCandidateResult] = []
    for index, candidate in enumerate(candidates, start=1):
        callback = status_callback if index == 1 else None
        if len(candidates) > 1 and status_callback is not None:
            status_callback(
                "model_candidate_started",
                {"candidate_index": index, "candidate_count": len(candidates), "candidate": candidate.name},
            )
        result = _evaluate_round_model_candidate(
            candles,
            strategy,
            objective,
            candidate,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            batch_size=batch_size,
            require_gpu=require_gpu,
            status_callback=callback,
        )
        if len(candidates) > 1 and status_callback is not None:
            status_callback(
                "model_candidate_complete",
                {
                    "candidate_index": index,
                    "candidate_count": len(candidates),
                    "candidate": candidate.name,
                    "score": float(result.score),
                },
        )
        if best is None or _round_candidate_rank_key(result) > _round_candidate_rank_key(best):
            best = result
        evaluated.append(replace(result, training_rows=[], selection_rows=[], rows=[], validation_rows=[]))
        gc.collect()
    if best is None:
        raise ValueError("no model candidates were evaluated")
    best_strategy = _strategy_for_round_candidate(strategy, best.candidate)
    best = _select_hybrid_model_zoo_if_accepted(
        best,
        best_strategy,
        objective,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        batch_size=batch_size,
        status_callback=status_callback,
    )
    best = _select_rule_alpha_model_zoo_if_accepted(
        best,
        best_strategy,
        objective,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        batch_size=batch_size,
        status_callback=status_callback,
    )
    best.model.model_candidate_count = len(candidates)
    selected_candidate_name = best.candidate.name
    if str(getattr(best.model, "model_family", "") or "") == "rule_alpha_model_zoo" and getattr(best.model, "rule_alpha_profile", ""):
        selected_candidate_name = f"{selected_candidate_name}+rule_alpha:{best.model.rule_alpha_profile}"
    best.model.model_selected_candidate = selected_candidate_name
    best.model.model_selection_score = float(best.score)
    diagnostic_results = [
        replace(best, training_rows=[], selection_rows=[], rows=[], validation_rows=[])
        if result.candidate == best.candidate
        else result
        for result in evaluated
    ]
    best.model.round_candidate_diagnostics = [
        _round_candidate_diagnostic(result, strategy=strategy, selected=result.candidate == best.candidate)
        for result in diagnostic_results
    ]
    if len(candidates) > 1 and status_callback is not None:
        status_callback(
            "model_candidate_search_complete",
            {
                "candidate_count": len(candidates),
                "selected_candidate": best.candidate.name,
                "selection_score": float(best.score),
                "selected_stop_loss_pct": float(best_strategy.stop_loss_pct),
                "selected_take_profit_pct": float(best_strategy.take_profit_pct),
                "selected_cooldown_minutes": int(best_strategy.cooldown_minutes),
            },
        )
    return best.model, best.report, list(best.rows), list(best.validation_rows)


def build_round_evidence(
    *,
    round_id: str,
    client: BinanceClient,
    strategy: StrategyConfig,
    quote_asset: str = "USDT",
    symbol_count: int = 50,
    symbols: Sequence[str] | None = None,
    interval: str = "15m",
    market_type: str = "spot",
    objective_name: str = "conservative",
    starting_cash: float = 1000.0,
    compute_backend: str = "auto",
    batch_size: int = 8192,
    data_root: Path = Path("data/optimization"),
    docs_root: Path = Path("docs/optimization"),
    db_path: Path = Path("data/market_data.sqlite"),
    require_prefilled_data: bool = False,
    min_data_rows: int = 0,
    min_coverage_ratio: float = 0.995,
    max_gap_count: int = 0,
    require_verified_checksum: bool = False,
    require_gpu: bool = False,
    promotion_grade: bool = False,
    min_promotion_data_years: float = 2.0,
    use_objective_strategy_defaults: bool = False,
    model_candidate_count: int = 1,
    data_start_ms: int | None = None,
    data_end_ms: int | None = None,
) -> dict[str, object]:
    promotion_grade = bool(promotion_grade)
    min_promotion_data_years = max(0.0, _finite(min_promotion_data_years, 0.0))
    min_coverage_ratio = max(0.0, min(1.0, _finite(min_coverage_ratio, 0.995)))
    max_gap_count = max(0, int(max_gap_count))
    min_data_rows = max(0, int(min_data_rows))
    if data_start_ms is not None and data_end_ms is not None and int(data_start_ms) > int(data_end_ms):
        raise ValueError("data_start_ms must be earlier than or equal to data_end_ms")
    if promotion_grade:
        interval = "1s"
        required_symbols = _promotion_required_symbols(quote_asset)
        if symbols:
            _validate_promotion_symbol_scope(symbols, quote_asset)
        symbols = required_symbols
        symbol_count = len(required_symbols)
        require_prefilled_data = True
        require_verified_checksum = True
        min_coverage_ratio = max(min_coverage_ratio, 0.995)
        max_gap_count = 0
        min_data_rows = max(min_data_rows, _promotion_min_rows(min_promotion_data_years, interval))
    if market_type == "futures" and str(interval) == "1s":
        if not require_prefilled_data:
            raise ValueError("futures 1s optimization requires prefilled aggTrades-derived candles")
        interval = "1s"
    else:
        interval = validate_interval(interval, market_type)
    paths = make_evidence_paths(round_id, data_root=data_root, docs_root=docs_root, market_db_path=db_path)
    for directory in (paths.output_dir, paths.docs_dir, paths.docs_data_dir, paths.docs_charts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    objective = get_objective(objective_name)
    evidence_strategy = (
        strategy_with_objective_defaults(strategy, objective)
        if use_objective_strategy_defaults
        else strategy
    )
    effective_leverage = effective_leverage_for_market(evidence_strategy, market_type)
    leverage_applies = market_type == "futures"
    backend_info = resolve_backend(effective_training_backend_name(compute_backend))
    if require_gpu and backend_info.kind == "cpu":
        raise ValueError(f"gpu_required_but_unavailable: {backend_info.reason or 'resolved to CPU'}")

    def write_status(phase: str, **payload: object) -> None:
        _write_round_status(
            paths,
            status=str(payload.pop("status", "running")),
            phase=phase,
            compute_backend_requested=backend_info.requested,
            compute_backend_kind=backend_info.kind,
            compute_backend_device=backend_info.device,
            compute_backend_reason=backend_info.reason,
            require_gpu=bool(require_gpu),
            **payload,
        )

    requested_symbols = [str(symbol).strip().upper() for symbol in symbols or () if str(symbol).strip()]
    write_status(
        "selection_started",
        symbol_count_requested=(len(requested_symbols) if requested_symbols else int(symbol_count)),
        completed_symbol_count=0,
        symbols=requested_symbols,
        explicit_symbol_mode=bool(requested_symbols),
        health_required=bool(require_prefilled_data or min_data_rows > 0 or require_verified_checksum),
    )
    health_required = bool(require_prefilled_data or min_data_rows > 0 or require_verified_checksum)
    selection_health_rejections: list[dict[str, object]] = []
    if symbols:
        selected = select_named_symbols(client, evidence_strategy, symbols, quote_asset=quote_asset)
    elif health_required:
        selected, selection_health_rejections = select_data_healthy_top_liquidity_symbols(
            client,
            evidence_strategy,
            quote_asset=quote_asset,
            count=symbol_count,
            market_type=market_type,
            interval=interval,
            db_path=paths.market_db_path,
            min_rows=min_data_rows,
            min_coverage_ratio=min_coverage_ratio,
            max_gap_count=max_gap_count,
            require_verified_checksum=require_verified_checksum,
            min_span_years=min_promotion_data_years if promotion_grade else 0.0,
            start_ms=data_start_ms,
            end_ms=data_end_ms,
        )
    else:
        selected = select_top_liquidity_symbols(client, evidence_strategy, quote_asset=quote_asset, count=symbol_count)
        if len(selected) < max(1, int(symbol_count)):
            raise ValueError(
                "strict_liquidity_selection_shortfall: "
                f"selected {len(selected)}/{max(1, int(symbol_count))} strict live-eligible symbols; "
                "lower symbol_count or improve liquidity gates explicitly instead of filling with research-tier assets"
            )
    write_json_atomic(paths.docs_data_dir / "selected-universe.json", [item.asdict() for item in selected], indent=2, sort_keys=True)
    write_status(
        "selection_complete",
        symbol_count_requested=len(selected),
        symbols=[item.symbol for item in selected],
        completed_symbol_count=0,
        current_symbol="",
    )
    metrics: list[BacktestEvidence] = []
    data_health: list[dict[str, object]] = []
    candidate_diagnostic_rows: list[dict[str, object]] = []
    portfolio_aggregate: dict[int, dict[str, float]] = {}
    timeline_fieldnames = (
        "round_id", "symbol", "objective", "timestamp_ms", "timestamp_utc",
        "strategy_equity", "baseline_equity", "strategy_drawdown", "baseline_drawdown",
        "quote_volume", "trade_count", "rolling_quote_volume_median",
        "rolling_trade_count_median", "clock_bucket", "clock_bucket_quote_volume_median",
        "clock_bucket_trade_count_median", "data_probed_low_session_flag",
        "low_liquidity_flag", "weekend_flag", "utc_hour", "utc_weekday",
    )
    for selected_index, item in enumerate(selected, start=1):
        write_status(
            "symbol_started",
            symbol_count_requested=len(selected),
            completed_symbol_count=len(metrics),
            current_symbol=item.symbol,
            current_symbol_index=selected_index,
        )
        candles: list[Candle] = []
        coverage = None
        try:
            if item.tier == "explicit-symbol" and item.reasons:
                reasons = ", ".join(item.reasons)
                raise ValueError(f"symbol_selection_failed: {reasons}")
            write_status(
                "data_health_started",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
            )
            health = market_data_health_for_symbol(
                db_path=paths.market_db_path,
                symbol=item.symbol,
                market_type=market_type,
                interval=interval,
                min_rows=min_data_rows,
                min_coverage_ratio=min_coverage_ratio,
                max_gap_count=max_gap_count,
                require_verified_checksum=require_verified_checksum,
                min_span_years=min_promotion_data_years if promotion_grade else 0.0,
                start_ms=data_start_ms,
                end_ms=data_end_ms,
            )
            data_health.append(health)
            write_status(
                "data_health_complete",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
                health_status=health.get("status"),
                rows=health.get("rows"),
                coverage_ratio=health.get("coverage_ratio"),
                gap_count=health.get("gap_count"),
            )
            if health_required and health.get("status") != "ok":
                reasons = ", ".join(str(reason) for reason in health.get("reasons", []) if reason)
                raise ValueError(f"data_health_failed: {reasons or 'unknown'}")
            write_status(
                "load_candles_started",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
            )
            candles = fetch_full_history(
                client,
                item.symbol,
                interval,
                db_path=paths.market_db_path,
                market_type=market_type,
                batch_size=batch_size,
                allow_network_backfill=not require_prefilled_data,
                start_ms=data_start_ms,
                end_ms=data_end_ms,
            )
            write_status(
                "load_candles_complete",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
                candle_count=len(candles),
            )
            coverage = describe_candle_coverage(
                symbol=item.symbol,
                market_type=market_type,
                interval=interval,
                available_candles=candles,
                used_candles=candles,
                rows_used=max(0, len(candles) - 1),
                requested_start_ms=data_start_ms,
                requested_end_ms=data_end_ms,
                source_scope=(
                    "binance_windowed_public_market_data"
                    if data_start_ms is not None or data_end_ms is not None
                    else "binance_full_history_public_market_data"
                ),
            )
            def symbol_train_status(phase: str, payload: Mapping[str, object]) -> None:
                write_status(
                    phase,
                    symbol_count_requested=len(selected),
                    completed_symbol_count=len(metrics),
                    current_symbol=item.symbol,
                    current_symbol_index=selected_index,
                    **dict(payload),
                )

            model, train_report, rows, validation_rows = train_round_model(
                candles,
                evidence_strategy,
                objective,
                market_type=market_type,
                starting_cash=starting_cash,
                compute_backend=compute_backend,
                batch_size=batch_size,
                require_gpu=require_gpu,
                model_candidate_count=model_candidate_count,
                status_callback=symbol_train_status,
            )
            symbol_candidate_diagnostics = _candidate_diagnostic_rows(item.symbol, model)
            candidate_diagnostic_rows.extend(symbol_candidate_diagnostics)
            selected_strategy = apply_model_strategy_overrides(evidence_strategy, model)
            selected_effective_leverage = effective_leverage_for_market(selected_strategy, market_type)
            write_status(
                "holdout_backtest_started",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
                validation_rows=len(validation_rows),
                selected_stop_loss_pct=float(selected_strategy.stop_loss_pct),
                selected_take_profit_pct=float(selected_strategy.take_profit_pct),
                selected_cooldown_minutes=int(selected_strategy.cooldown_minutes),
            )
            result = run_backtest(
                validation_rows,
                model,
                selected_strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=compute_backend,
                score_batch_size=batch_size,
                symbol_profile=SymbolExecutionProfile(
                    item.symbol,
                    item.spread_bps,
                    item.quote_volume,
                    item.trade_count,
                    item.liquidity_score,
                    latency_ms=selected_strategy.latency_buffer_ms,
                    liquidity_haircut=selected_strategy.testnet_liquidity_haircut,
                ),
            )
            if require_gpu:
                _require_non_cpu_backend(
                    result.scoring_backend_kind,
                    result.scoring_backend_reason,
                    "holdout_scoring",
                )
            write_status(
                "holdout_backtest_complete",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
                realized_pnl=float(result.realized_pnl),
                max_drawdown=float(result.max_drawdown),
                closed_trades=int(result.closed_trades),
                stopped_by_liquidation=bool(getattr(result, "stopped_by_liquidation", False)),
                liquidation_events=int(getattr(result, "liquidation_events", 0)),
                scoring_backend_kind=result.scoring_backend_kind,
                scoring_backend_device=result.scoring_backend_device,
            )
            market_edge = build_market_edge_report(result, objective)
            selection_gate_passed = bool(getattr(model, "round_selection_gate_passed", True))
            accepted = bool(
                selection_gate_passed
                and objective.accepts(result)
                and market_edge.accepted
                and not result.stopped_by_drawdown
                and not bool(getattr(result, "stopped_by_liquidation", False))
                and int(getattr(result, "liquidation_events", 0)) <= 0
            )
            base_reason = objective.reject_reason(result) or market_edge.reason
            if not selection_gate_passed:
                selection_reason = str(getattr(model, "round_selection_reject_reason", "") or "selection_gate_failed")
                reason = f"selection_gate_failed: {selection_reason}"
                if base_reason:
                    reason = f"{reason}; {base_reason}"
            else:
                reason = base_reason
            strategy_points = _result_points(result)
            baseline_series = _baseline_equity_series(validation_rows, starting_cash, selected_strategy, market_type=market_type)
            baseline_points = [
                EquityPoint(index=index, equity=_finite(point.get("equity")), drawdown=_finite(point.get("drawdown")), timestamp_ms=int(point["timestamp"]))
                for index, point in enumerate(baseline_series)
                if "timestamp" in point
            ]
            write_status(
                "artifact_stream_started",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
                validation_rows=len(validation_rows),
            )
            validation_timestamps = {
                int(getattr(row, "timestamp", 0))
                for row in validation_rows
                if int(getattr(row, "timestamp", 0)) > 0
            }
            liquidity_by_close = _rolling_liquidity_flags(candles, timestamps=validation_timestamps)
            point_by_timestamp = {int(point.timestamp_ms or 0): point for point in strategy_points}
            baseline_by_timestamp = {int(point.timestamp_ms or 0): point for point in baseline_points}
            symbol_timeline_path = paths.docs_data_dir / f"{item.rank:02d}-{item.symbol}-{objective.name}-timeline.csv.gz"
            symbol_timeline_path.parent.mkdir(parents=True, exist_ok=True)
            timeline_count = 0
            low_liquidity_count = 0
            weekend_count = 0
            with _open_text_writer(symbol_timeline_path) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(timeline_fieldnames))
                writer.writeheader()
                for timestamp, point in sorted(point_by_timestamp.items()):
                    if timestamp <= 0:
                        continue
                    baseline = baseline_by_timestamp.get(timestamp)
                    liquidity = liquidity_by_close.get(timestamp, {})
                    row = {
                        "round_id": round_id,
                        "symbol": item.symbol,
                        "objective": objective.name,
                        "timestamp_ms": timestamp,
                        "timestamp_utc": iso_utc(timestamp),
                        "strategy_equity": point.equity,
                        "baseline_equity": baseline.equity if baseline else "",
                        "strategy_drawdown": point.drawdown,
                        "baseline_drawdown": baseline.drawdown if baseline else "",
                        "quote_volume": liquidity.get("quote_volume", ""),
                        "trade_count": liquidity.get("trade_count", ""),
                        "rolling_quote_volume_median": liquidity.get("rolling_quote_volume_median", ""),
                        "rolling_trade_count_median": liquidity.get("rolling_trade_count_median", ""),
                        "clock_bucket": liquidity.get("clock_bucket", ""),
                        "clock_bucket_quote_volume_median": liquidity.get("clock_bucket_quote_volume_median", ""),
                        "clock_bucket_trade_count_median": liquidity.get("clock_bucket_trade_count_median", ""),
                        "data_probed_low_session_flag": liquidity.get("data_probed_low_session_flag", ""),
                        "low_liquidity_flag": liquidity.get("low_liquidity_flag", ""),
                        "weekend_flag": liquidity.get("weekend_flag", ""),
                        "utc_hour": liquidity.get("utc_hour", ""),
                        "utc_weekday": liquidity.get("utc_weekday", ""),
                    }
                    writer.writerow(row)
                    _update_portfolio_aggregate(portfolio_aggregate, row)
                    timeline_count += 1
                    if str(row.get("low_liquidity_flag")).lower() == "true":
                        low_liquidity_count += 1
                    if str(row.get("weekend_flag")).lower() == "true":
                        weekend_count += 1
            chart_path = paths.docs_charts_dir / f"{item.rank:02d}-{item.symbol}-{objective.name}.svg"
            chart_path.write_text(
                render_comparison_svg(strategy_points, baseline_points, title=f"{item.symbol} {objective.label} Backtest vs Passive Baseline"),
                encoding="utf-8",
            )
            write_status(
                "artifact_stream_complete",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
                timeline_rows=timeline_count,
                timeline_csv_path=str(symbol_timeline_path).replace("\\", "/"),
                chart_path=str(chart_path).replace("\\", "/"),
                chart_bytes=chart_path.stat().st_size if chart_path.exists() else 0,
            )
            low_rate = low_liquidity_count / timeline_count if timeline_count else 0.0
            weekend_rate = weekend_count / timeline_count if timeline_count else 0.0
            metric = BacktestEvidence(
                round_id=round_id,
                symbol=item.symbol,
                objective=objective.name,
                risk_level=str(selected_strategy.risk_level),
                leverage=float(selected_strategy.leverage),
                effective_leverage=float(selected_effective_leverage),
                leverage_applies=bool(leverage_applies),
                risk_per_trade=float(selected_strategy.risk_per_trade),
                max_position_pct=float(selected_strategy.max_position_pct),
                max_drawdown_limit_pct=float(selected_strategy.max_drawdown_limit * 100.0),
                accepted=accepted,
                reason=reason,
                start_utc=iso_utc(int(validation_rows[0].timestamp)) if validation_rows else coverage.used_start_utc,
                end_utc=iso_utc(int(validation_rows[-1].timestamp)) if validation_rows else coverage.used_end_utc,
                duration_years=(
                    (int(validation_rows[-1].timestamp) - int(validation_rows[0].timestamp)) / (365.25 * 24 * 60 * 60 * 1000)
                    if len(validation_rows) >= 2 else 0.0
                ),
                candles=len(candles),
                rows=len(validation_rows),
                training_rows=int(getattr(train_report, "row_count", 0) or 0),
                training_positive_rate_pct=float(getattr(train_report, "positive_rate", 0.0) or 0.0) * 100.0,
                model_candidate_count=int(getattr(model, "model_candidate_count", 1) or 1),
                model_selected_candidate=str(getattr(model, "model_selected_candidate", "default") or "default"),
                model_selection_score=(
                    float(getattr(model, "model_selection_score"))
                    if getattr(model, "model_selection_score", None) is not None
                    else None
                ),
                candidate_diagnostics_path=str(paths.candidate_diagnostics_csv_path).replace("\\", "/"),
                model_training_backend_kind=str(getattr(model, "training_backend_kind", "")),
                model_training_backend_device=str(getattr(model, "training_backend_device", "")),
                probability_calibration_backend_kind=str(
                    getattr(model, "probability_calibration_backend_kind", "")
                ),
                probability_calibration_backend_device=str(
                    getattr(model, "probability_calibration_backend_device", "")
                ),
                threshold_source=(
                    str(getattr(model, "threshold_source"))
                    if getattr(model, "threshold_source", None) is not None
                    else None
                ),
                threshold_calibration_score=(
                    float(getattr(model, "threshold_calibration_score"))
                    if getattr(model, "threshold_calibration_score", None) is not None
                    else None
                ),
                threshold_calibration_pnl=(
                    float(getattr(model, "threshold_calibration_pnl"))
                    if getattr(model, "threshold_calibration_pnl", None) is not None
                    else None
                ),
                threshold_calibration_trades=int(getattr(model, "threshold_calibration_trades", 0) or 0),
                threshold_diagnostic_best_threshold=(
                    float(getattr(model, "threshold_diagnostic_best_threshold"))
                    if getattr(model, "threshold_diagnostic_best_threshold", None) is not None
                    else None
                ),
                threshold_diagnostic_best_score=(
                    float(getattr(model, "threshold_diagnostic_best_score"))
                    if getattr(model, "threshold_diagnostic_best_score", None) is not None
                    else None
                ),
                threshold_diagnostic_best_pnl=(
                    float(getattr(model, "threshold_diagnostic_best_pnl"))
                    if getattr(model, "threshold_diagnostic_best_pnl", None) is not None
                    else None
                ),
                threshold_diagnostic_best_trades=int(
                    getattr(model, "threshold_diagnostic_best_trades", 0) or 0
                ),
                decision_threshold=(
                    float(getattr(model, "decision_threshold"))
                    if getattr(model, "decision_threshold", None) is not None
                    else None
                ),
                long_decision_threshold=(
                    float(getattr(model, "long_decision_threshold"))
                    if getattr(model, "long_decision_threshold", None) is not None
                    else None
                ),
                short_decision_threshold=(
                    float(getattr(model, "short_decision_threshold"))
                    if getattr(model, "short_decision_threshold", None) is not None
                    else None
                ),
                threshold_diagnostic_best_long_threshold=(
                    float(getattr(model, "threshold_diagnostic_best_long_threshold"))
                    if getattr(model, "threshold_diagnostic_best_long_threshold", None) is not None
                    else None
                ),
                threshold_diagnostic_best_short_threshold=(
                    float(getattr(model, "threshold_diagnostic_best_short_threshold"))
                    if getattr(model, "threshold_diagnostic_best_short_threshold", None) is not None
                    else None
                ),
                round_selection_gate_passed=bool(getattr(model, "round_selection_gate_passed", True)),
                round_selection_reject_reason=(
                    str(getattr(model, "round_selection_reject_reason"))
                    if getattr(model, "round_selection_reject_reason", None)
                    else None
                ),
                model_quality_warnings="; ".join(str(item) for item in getattr(model, "quality_warnings", []) or []),
                meta_label_policy_reason=(
                    str(getattr(model, "meta_label_policy", {}).get("reason"))
                    if isinstance(getattr(model, "meta_label_policy", None), dict)
                    and getattr(model, "meta_label_policy", {}).get("reason")
                    else None
                ),
                starting_cash=float(result.starting_cash),
                ending_cash=float(result.ending_cash),
                realized_pnl=float(result.realized_pnl),
                roi_pct=float(result.realized_pnl / max(1.0, result.starting_cash) * 100.0),
                buy_hold_pnl=float(result.buy_hold_pnl),
                buy_hold_roi_pct=float(result.buy_hold_pnl / max(1.0, result.starting_cash) * 100.0),
                edge_vs_buy_hold=float(result.edge_vs_buy_hold),
                market_edge_pct=float(market_edge.net_edge_pct * 100.0),
                max_drawdown_pct=float(result.max_drawdown * 100.0),
                trades=int(result.trades),
                closed_trades=int(result.closed_trades),
                win_rate_pct=float(result.win_rate * 100.0),
                stopped_by_liquidation=bool(getattr(result, "stopped_by_liquidation", False)),
                liquidation_events=int(getattr(result, "liquidation_events", 0)),
                liquidation_loss=float(getattr(result, "liquidation_loss", 0.0)),
                total_fees=float(result.total_fees),
                profit_factor=float(result.profit_factor),
                expectancy=float(result.expectancy),
                avg_trade_return_pct=float(result.average_trade_return * 100.0),
                max_consecutive_losses=int(result.max_consecutive_losses),
                low_liquidity_sample_rate_pct=float(low_rate * 100.0),
                weekend_sample_rate_pct=float(weekend_rate * 100.0),
                scoring_backend_kind=str(result.scoring_backend_kind),
                scoring_backend_device=str(result.scoring_backend_device),
                chart_path=str(chart_path).replace("\\", "/"),
                timeline_csv_path=str(symbol_timeline_path).replace("\\", "/"),
            )
        except (BinanceAPIError, OSError, ValueError, RuntimeError) as exc:
            metric = BacktestEvidence(
                round_id=round_id,
                symbol=item.symbol,
                objective=objective.name,
                risk_level=str(evidence_strategy.risk_level),
                leverage=float(evidence_strategy.leverage),
                effective_leverage=float(effective_leverage),
                leverage_applies=bool(leverage_applies),
                risk_per_trade=float(evidence_strategy.risk_per_trade),
                max_position_pct=float(evidence_strategy.max_position_pct),
                max_drawdown_limit_pct=float(evidence_strategy.max_drawdown_limit * 100.0),
                accepted=False,
                reason=str(exc)[:240],
                start_utc=(coverage.used_start_utc if coverage is not None else None),
                end_utc=(coverage.used_end_utc if coverage is not None else None),
                duration_years=float(coverage.used_duration_years if coverage is not None else 0.0),
                candles=len(candles),
                rows=0,
                training_rows=0,
                training_positive_rate_pct=0.0,
                model_candidate_count=int(max(1, model_candidate_count)),
                model_selected_candidate="",
                model_selection_score=None,
                candidate_diagnostics_path="",
                model_training_backend_kind="error",
                model_training_backend_device="error",
                probability_calibration_backend_kind="error",
                probability_calibration_backend_device="error",
                threshold_source=None,
                threshold_calibration_score=None,
                threshold_calibration_pnl=None,
                threshold_calibration_trades=0,
                threshold_diagnostic_best_threshold=None,
                threshold_diagnostic_best_score=None,
                threshold_diagnostic_best_pnl=None,
                threshold_diagnostic_best_trades=0,
                decision_threshold=None,
                long_decision_threshold=None,
                short_decision_threshold=None,
                threshold_diagnostic_best_long_threshold=None,
                threshold_diagnostic_best_short_threshold=None,
                round_selection_gate_passed=False,
                round_selection_reject_reason=str(exc)[:240],
                model_quality_warnings="",
                meta_label_policy_reason=None,
                starting_cash=float(starting_cash),
                ending_cash=float(starting_cash),
                realized_pnl=0.0,
                roi_pct=0.0,
                buy_hold_pnl=0.0,
                buy_hold_roi_pct=0.0,
                edge_vs_buy_hold=0.0,
                market_edge_pct=0.0,
                max_drawdown_pct=0.0,
                trades=0,
                closed_trades=0,
                win_rate_pct=0.0,
                total_fees=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                avg_trade_return_pct=0.0,
                max_consecutive_losses=0,
                low_liquidity_sample_rate_pct=0.0,
                weekend_sample_rate_pct=0.0,
                scoring_backend_kind="error",
                scoring_backend_device="error",
                chart_path="",
                timeline_csv_path="",
            )
        metrics.append(metric)
        write_status(
            "symbol_completed",
            symbol_count_requested=len(selected),
            completed_symbol_count=len(metrics),
            current_symbol="",
            last_symbol=item.symbol,
            last_symbol_accepted=metric.accepted,
            last_symbol_reason=metric.reason,
        )
        candles = []
        rows = []
        validation_rows = []
        gc.collect()
    metric_rows = [metric.asdict() for metric in metrics]
    _write_csv(
        paths.metrics_csv_path,
        metric_rows,
        tuple(metric_rows[0].keys()) if metric_rows else ("round_id", "symbol", "objective"),
    )
    candidate_diagnostic_fieldnames = (
        "symbol", "candidate_index", "name", "selected", "score", "signal_threshold",
        "stop_loss_pct", "take_profit_pct", "cooldown_minutes", "min_position_hold_bars",
        "flat_signal_exit_grace_bars", "label_threshold", "label_lookahead", "label_mode",
        "focal_gamma", "model_family", "probability_inverted", "hybrid_profile", "hybrid_base_score",
        "hybrid_best_score", "hybrid_evaluated_profiles", "hybrid_expert_count",
        "rule_alpha_profile", "rule_alpha_family", "rule_alpha_best_score",
        "rule_alpha_best_pnl", "rule_alpha_best_closed_trades",
        "rule_alpha_best_win_rate", "rule_alpha_best_profit_factor",
        "rule_alpha_best_max_drawdown", "rule_alpha_best_exit_reason_counts",
        "rule_alpha_best_side_counts",
        "rule_alpha_best_reject_reason", "rule_alpha_probability_inverted",
        "rule_alpha_evaluated_candidates",
        "rule_alpha_active_candidates", "rule_alpha_profitable_candidates",
        "rule_alpha_accepted_candidates", "rule_alpha_max_closed_trades",
        "rule_alpha_event_candidates_with_signals", "rule_alpha_event_positive_candidates",
        "rule_alpha_event_best_candidate", "rule_alpha_event_best_net_edge_bps",
        "rule_alpha_event_best_signal_count", "rule_alpha_event_best_hit_rate",
        "rule_alpha_event_best_horizon_bars", "rule_alpha_event_best_probability_inverted",
        "rule_alpha_most_active_candidate", "rule_alpha_most_active_pnl",
        "rule_alpha_most_active_profit_factor", "rule_alpha_most_active_reject_reason",
        "rule_alpha_best_pnl_candidate", "rule_alpha_best_pnl_candidate_pnl",
        "rule_alpha_best_pnl_candidate_closed_trades",
        "rule_alpha_families_with_trades", "rule_alpha_profiles_with_trades",
        "round_selection_gate_passed",
        "round_selection_reject_reason", "threshold_source", "decision_threshold",
        "long_decision_threshold", "short_decision_threshold",
        "threshold_calibration_score", "threshold_calibration_pnl", "threshold_calibration_trades",
        "threshold_baseline_score", "threshold_baseline_pnl", "threshold_baseline_trades",
        "threshold_min_closed_trades", "threshold_min_trades_per_day",
        "threshold_selected_trades_per_day", "threshold_best_trades_per_day",
        "threshold_evaluated_thresholds",
        "threshold_diagnostic_best_threshold", "threshold_diagnostic_best_long_threshold",
        "threshold_diagnostic_best_short_threshold", "threshold_diagnostic_best_score",
        "threshold_diagnostic_best_pnl", "threshold_diagnostic_best_trades",
        "selection_realized_pnl", "selection_closed_trades", "selection_max_drawdown",
        "selection_win_rate", "selection_total_fees", "selection_profit_factor",
        "selection_expectancy", "selection_edge_vs_buy_hold", "selection_trades_per_day",
        "selection_trades_per_day_cap_hit", "selection_regime_entry_skips",
        "selection_meta_label_skips", "selection_meta_label_downsizes", "selection_liquidation_events",
        "selection_reject_reason",
    )
    _write_csv(
        paths.candidate_diagnostics_csv_path,
        candidate_diagnostic_rows,
        candidate_diagnostic_fieldnames,
    )
    write_json_atomic(
        paths.candidate_diagnostics_json_path,
        candidate_diagnostic_rows,
        indent=2,
        sort_keys=True,
    )
    portfolio_rows = _portfolio_timeline_from_aggregate(portfolio_aggregate)
    _write_csv(
        paths.timeline_csv_path,
        portfolio_rows,
        (
            "timestamp_ms", "timestamp_utc", "symbols_reporting", "mean_strategy_equity",
            "mean_baseline_equity", "mean_drawdown", "low_liquidity_symbol_count",
        ),
    )
    accepted_metrics = [metric for metric in metrics if metric.accepted]
    critical_analysis = critical_round_analysis(metrics)
    tracked_artifacts = [
        str(paths.progress_csv_path).replace("\\", "/"),
        str(paths.metrics_csv_path).replace("\\", "/"),
        str(paths.candidate_diagnostics_csv_path).replace("\\", "/"),
        str(paths.candidate_diagnostics_json_path).replace("\\", "/"),
        str(paths.timeline_csv_path).replace("\\", "/"),
        str(paths.report_path).replace("\\", "/"),
        str(paths.data_health_path).replace("\\", "/"),
        str(paths.status_path).replace("\\", "/"),
        str(paths.docs_data_dir / "selected-universe.json").replace("\\", "/"),
    ]
    for metric in metrics:
        if metric.chart_path:
            tracked_artifacts.append(metric.chart_path)
        if metric.timeline_csv_path:
            tracked_artifacts.append(metric.timeline_csv_path)
    progress = {
        "round_id": round_id,
        "generated_at_utc": _utc_now(),
        "symbol_count": len(metrics),
        "accepted_symbol_count": len(accepted_metrics),
        "mean_roi_pct": statistics.mean([metric.roi_pct for metric in metrics]) if metrics else 0.0,
        "median_roi_pct": statistics.median([metric.roi_pct for metric in metrics]) if metrics else 0.0,
        "mean_baseline_roi_pct": statistics.mean([metric.buy_hold_roi_pct for metric in metrics]) if metrics else 0.0,
        "mean_market_edge_pct": statistics.mean([metric.market_edge_pct for metric in metrics]) if metrics else 0.0,
        "median_market_edge_pct": statistics.median([metric.market_edge_pct for metric in metrics]) if metrics else 0.0,
        "worst_max_drawdown_pct": max([metric.max_drawdown_pct for metric in metrics], default=0.0),
        "total_trades": sum(metric.trades for metric in metrics),
        "total_closed_trades": sum(metric.closed_trades for metric in metrics),
        "zero_trade_symbol_count": int(critical_analysis["zero_trade_symbol_count"]),
        "profitable_symbol_count": int(critical_analysis["profitable_symbol_count"]),
        "critical_verdict": str(critical_analysis["verdict"]),
        "mean_low_liquidity_sample_rate_pct": statistics.mean([metric.low_liquidity_sample_rate_pct for metric in metrics]) if metrics else 0.0,
    }
    _write_csv(paths.progress_csv_path, [progress], tuple(progress.keys()))
    write_json_atomic(paths.data_health_path, data_health, indent=2, sort_keys=True)
    promotion_contract = (
        promotion_grade_contract(
            market_type=market_type,
            quote_asset=quote_asset,
            interval=interval,
            selected_symbols=[item.symbol for item in selected],
            data_health=data_health,
            critical_analysis=critical_analysis,
            require_prefilled_data=require_prefilled_data,
            require_verified_checksum=require_verified_checksum,
            min_data_rows=min_data_rows,
            min_data_years=min_promotion_data_years,
            min_coverage_ratio=min_coverage_ratio,
            max_gap_count=max_gap_count,
            require_gpu=require_gpu,
            backend_kind=backend_info.kind,
        )
        if promotion_grade
        else {
            "status": "not_requested",
            "reasons": [],
            "required_symbols": _promotion_required_symbols(quote_asset),
            "selected_symbols": [item.symbol for item in selected],
        }
    )
    write_status(
        "round_complete",
        status="complete",
        symbol_count_requested=len(selected),
        completed_symbol_count=len(metrics),
        accepted_symbol_count=len(accepted_metrics),
        metrics_csv_path=str(paths.metrics_csv_path).replace("\\", "/"),
        report_path=str(paths.report_path).replace("\\", "/"),
    )
    tracked_artifacts = sorted(dict.fromkeys(tracked_artifacts))
    artifact_integrity = _artifact_integrity_manifest(
        tracked_artifacts,
        report_path=paths.report_path,
    )
    report = {
        "round_id": round_id,
        "generated_at_utc": _utc_now(),
        "artifact_class": "exchange_sourced_backtest_graph_data",
        "evidence_verdict": str(critical_analysis["verdict"]),
        "critical_analysis": critical_analysis,
        "tracked_repo_artifact": True,
        "data_source": "Binance public market data stored in SQLite",
        "market_db_path": str(paths.market_db_path).replace("\\", "/"),
        "market_type": market_type,
        "quote_asset": quote_asset,
        "interval": interval,
        "objective": objective.name,
        "promotion_grade": bool(promotion_grade),
        "promotion_grade_contract": promotion_contract,
        "min_promotion_data_years": float(min_promotion_data_years),
        "use_objective_strategy_defaults": bool(use_objective_strategy_defaults),
        "strategy": evidence_strategy.asdict(),
        "configured_leverage": float(evidence_strategy.leverage),
        "effective_leverage": float(effective_leverage),
        "leverage_applies": bool(leverage_applies),
        "compute_backend_requested": backend_info.requested,
        "compute_backend_kind": backend_info.kind,
        "compute_backend_device": backend_info.device,
        "compute_backend_reason": backend_info.reason,
        "require_gpu": bool(require_gpu),
        "model_candidate_count": int(max(1, model_candidate_count)),
        "starting_cash": float(starting_cash),
        "symbol_count_requested": len([str(symbol).strip() for symbol in (symbols or []) if str(symbol).strip()]) if symbols else int(symbol_count),
        "symbol_count_completed": len(metrics),
        "explicit_symbols": [str(symbol).upper() for symbol in (symbols or [])],
        "require_prefilled_data": bool(require_prefilled_data),
        "min_data_rows": int(min_data_rows),
        "min_coverage_ratio": float(min_coverage_ratio),
        "max_gap_count": int(max_gap_count),
        "require_verified_checksum": bool(require_verified_checksum),
        "data_start_ms": data_start_ms,
        "data_end_ms": data_end_ms,
        "data_start_utc": iso_utc(data_start_ms),
        "data_end_utc": iso_utc(data_end_ms),
        "health_filtered_symbol_selection": bool(not symbols and health_required),
        "selection_health_rejections": selection_health_rejections,
        "data_health_path": str(paths.data_health_path).replace("\\", "/"),
        "status_path": str(paths.status_path).replace("\\", "/"),
        "data_health": data_health,
        "selected_universe_path": str(paths.docs_data_dir / "selected-universe.json").replace("\\", "/"),
        "metrics_csv_path": str(paths.metrics_csv_path).replace("\\", "/"),
        "candidate_diagnostics_csv_path": str(paths.candidate_diagnostics_csv_path).replace("\\", "/"),
        "candidate_diagnostics_json_path": str(paths.candidate_diagnostics_json_path).replace("\\", "/"),
        "portfolio_timeline_csv_path": str(paths.timeline_csv_path).replace("\\", "/"),
        "progress_csv_path": str(paths.progress_csv_path).replace("\\", "/"),
        "progress": progress,
        "metrics": metric_rows,
        "candidate_diagnostics": candidate_diagnostic_rows,
        "tracked_artifacts": tracked_artifacts,
        "artifact_integrity": artifact_integrity,
    }
    write_json_atomic(paths.report_path, report, indent=2, sort_keys=True)
    return report


__all__ = [
    "BacktestEvidence",
    "EvidencePaths",
    "SelectedSymbol",
    "build_round_evidence",
    "critical_round_analysis",
    "fetch_full_history",
    "effective_leverage_for_market",
    "make_evidence_paths",
    "market_data_health_for_symbol",
    "parse_evidence_timestamp_ms",
    "promotion_grade_contract",
    "render_comparison_svg",
    "select_data_healthy_top_liquidity_symbols",
    "select_named_symbols",
    "select_top_liquidity_symbols",
    "strategy_with_objective_defaults",
]
