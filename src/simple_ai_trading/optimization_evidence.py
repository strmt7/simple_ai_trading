"""Reproducible optimization-round evidence and graph-data generation."""

from __future__ import annotations

import csv
import copy
import gc
import gzip
import math
import statistics
from bisect import bisect_left, insort
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .advanced_model import (
    AdvancedFeatureConfig,
    advanced_feature_dimension,
    default_config_for,
    make_advanced_rows,
    train_advanced,
)
from .api import BinanceAPIError, BinanceClient, Candle
from .assets import MAX_AUTONOMOUS_LEVERAGE, is_supported_major_symbol
from .backtest import BacktestResult, calibrate_threshold_for_backtest, run_backtest
from .compute import resolve_backend
from .data_coverage import describe_candle_coverage, iso_utc
from .data_downloader import MarketDataSyncConfig, sync_market_data
from .execution_simulation import SymbolExecutionProfile
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
from .storage import write_json_atomic
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
    if negative_roi_symbols:
        warnings.append("some_symbols_negative_roi")
    if zero_trade_symbols:
        warnings.append("some_symbols_zero_closed_trades")
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


@dataclass(frozen=True)
class RoundModelCandidateResult:
    candidate: RoundModelCandidate
    score: float
    model: object
    report: object
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


def effective_leverage_for_market(strategy: StrategyConfig, market_type: str) -> float:
    """Return the leverage that can actually affect fills for the market type."""

    if str(market_type).lower() != "futures":
        return 1.0
    return max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, _finite(strategy.leverage, 1.0)))


def fetch_full_history(
    client: BinanceClient,
    symbol: str,
    interval: str,
    *,
    db_path: Path,
    market_type: str = "spot",
    batch_size: int = 1000,
    allow_network_backfill: bool = True,
) -> list[Candle]:
    with MarketDataStore(db_path) as store:
        candles = store.fetch_candles(symbol, market_type, interval)
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
        candles = store.fetch_candles(symbol, market_type, interval)
        quality = store.coverage_quality(symbol, market_type, interval, interval_milliseconds(interval))
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
) -> dict[str, object]:
    """Return a fail-closed health report for one optimization data series."""

    with MarketDataStore(db_path) as store:
        quality = store.coverage_quality(symbol, market_type, interval, interval_milliseconds(interval))
        archives = store.archive_files(symbol=symbol, market_type=market_type, interval=interval)
    archive_status_counts = _count_by(archives, "status")
    checksum_status_counts = _count_by(archives, "checksum_status")
    min_rows = max(0, int(min_rows))
    max_gap_count = max(0, int(max_gap_count))
    min_coverage_ratio = max(0.0, min(1.0, float(min_coverage_ratio)))
    reasons: list[str] = []
    warnings: list[str] = []
    if quality.coverage.count < min_rows:
        reasons.append(f"rows_below_min:{quality.coverage.count}/{min_rows}")
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
        )
        if health.get("status") == "ok":
            selected.append(replace(item, rank=len(selected) + 1))
        else:
            health_rejections.append({
                "selection_rank": int(item.rank),
                "symbol": item.symbol,
                "tier": item.tier,
                "rows": int(health.get("rows") or 0),
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


def _baseline_equity_series(rows: Sequence[object], starting_cash: float, cfg: StrategyConfig, *, market_type: str) -> list[dict[str, float | int]]:
    if not rows:
        return []
    first = _finite(getattr(rows[0], "close", 0.0))
    if first <= 0.0:
        return []
    # Use the same conservative risk notional convention as backtest._buy_hold_pnl.
    notional_pct = max(0.0, min(1.0, float(cfg.risk_per_trade) / max(1e-9, float(cfg.stop_loss_pct))))
    if market_type == "spot":
        notional_pct = min(notional_pct, float(cfg.max_position_pct))
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
    min_signal_threshold = {
        "conservative": 0.56,
        "regular": 0.52,
        "aggressive": 0.50,
    }.get(objective.name, 0.52)
    raw: list[tuple[str, float, float, float, float, float, str, float]] = [
        ("default", 1.0, 1.0, 1.0, 1.0, 1.0, str(base_feature_cfg.label_mode), 0.0),
        ("lower_lr_more_l2", 0.75, 0.75, 3.0, 1.20, 1.25, str(base_feature_cfg.label_mode), 0.0),
        ("short_horizon_forward", 0.50, 1.0, 1.0, 0.75, 0.75, "forward_return", 0.0),
        ("triple_barrier_base", 1.0, 0.90, 1.5, 1.0, 1.0, "triple_barrier", 0.0),
        ("triple_barrier_conservative", 0.75, 0.75, 3.0, 1.25, 1.50, "triple_barrier", 0.0),
        ("long_horizon_forward", 1.0, 0.75, 2.0, 1.40, 1.75, "forward_return", 0.0),
        ("lower_signal_short_forward", 0.65, 1.0, 1.25, 0.70, 0.60, "forward_return", -0.06),
        ("lower_signal_triple_barrier", 0.80, 0.90, 2.0, 0.80, 0.80, "triple_barrier", -0.06),
        ("frequency_probe_forward", 0.50, 1.10, 1.0, 0.55, 0.50, "forward_return", -0.10),
        ("high_conviction_triple_barrier", 1.0, 0.80, 3.0, 1.10, 1.25, "triple_barrier", 0.04),
    ]
    output: list[RoundModelCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for name, epoch_mult, lr_mult, l2_mult, threshold_mult, lookahead_mult, label_mode, signal_offset in raw:
        feature_cfg = replace(
            base_feature_cfg,
            label_threshold=max(1e-8, base_label_threshold * float(threshold_mult)),
            label_lookahead=max(1, int(round(base_label_lookahead * float(lookahead_mult)))),
            label_mode=str(label_mode),
            label_stop_threshold=(
                max(1e-8, float(base_feature_cfg.label_stop_threshold) * float(threshold_mult))
                if base_feature_cfg.label_stop_threshold is not None
                else None
            ),
        )
        candidate = RoundModelCandidate(
            name=name,
            feature_cfg=feature_cfg,
            epochs=max(1, int(round(base_epochs * float(epoch_mult)))),
            learning_rate=max(1e-6, base_lr * float(lr_mult)),
            l2_penalty=max(0.0, base_l2 * float(l2_mult)),
            signal_threshold=min(0.95, max(min_signal_threshold, base_threshold + float(signal_offset))),
        )
        key = (
            candidate.epochs,
            round(candidate.learning_rate, 12),
            round(candidate.l2_penalty, 12),
            round(candidate.signal_threshold, 12),
            advanced_feature_dimension(candidate.feature_cfg),
            candidate.feature_cfg.label_threshold,
            candidate.feature_cfg.label_lookahead,
            candidate.feature_cfg.label_mode,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
        if len(output) >= max(1, int(requested)):
            break
    return output


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
    if status_callback is not None:
        status_callback("feature_generation_started", {"candle_count": len(candles)})
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
    model.decision_threshold = float(candidate.signal_threshold)
    model.threshold_source = "objective_round_evidence_default"
    threshold_report = calibrate_threshold_for_backtest(
        selection_rows,
        model,
        strategy,
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
        model.threshold_source = "round_selection_backtest"
        model.threshold_calibration_score = float(threshold_report.score)
        model.threshold_calibration_pnl = float(threshold_report.realized_pnl)
        model.threshold_calibration_trades = int(threshold_report.closed_trades)
    base_result = run_backtest(
        selection_rows,
        model,
        strategy,
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
        strategy,
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
    if inverted_score > base_score + 1e-12:
        model = inverted_model
        score = inverted_score
        model.round_selection_gate_passed = True
        model.round_selection_reject_reason = ""
    elif base_accepts:
        model.round_selection_gate_passed = True
        model.round_selection_reject_reason = ""
    elif not math.isfinite(base_score):
        if inverted_raw_score > base_raw_score + 1e-12:
            model = inverted_model
            score = inverted_raw_score
            chosen_reject_reason = inverted_reject_reason
        else:
            score = base_raw_score
            chosen_reject_reason = base_reject_reason
        model.round_selection_gate_passed = False
        model.round_selection_reject_reason = str(chosen_reject_reason or "selection_gate_failed")
        if diagnostic_trades > 0:
            model.decision_threshold = diagnostic_threshold
            model.threshold_source = "round_selection_rejected_best_threshold_diagnostic"
            model.threshold_calibration_score = diagnostic_score
            model.threshold_calibration_pnl = diagnostic_pnl
            model.threshold_calibration_trades = diagnostic_trades
        else:
            model.threshold_source = "round_selection_rejected_diagnostic_holdout"
        model.quality_warnings = [
            *list(getattr(model, "quality_warnings", [])),
            "round_selection_gate_failed_diagnostic_holdout_only",
        ]
    return RoundModelCandidateResult(
        candidate=candidate,
        score=float(score),
        model=model,
        report=report,
        rows=list(rows),
        validation_rows=list(validation_rows),
    )


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
        if best is None or result.score > best.score + 1e-12:
            best = result
    if best is None:
        raise ValueError("no model candidates were evaluated")
    best.model.model_candidate_count = len(candidates)
    best.model.model_selected_candidate = best.candidate.name
    best.model.model_selection_score = float(best.score)
    if len(candidates) > 1 and status_callback is not None:
        status_callback(
            "model_candidate_search_complete",
            {
                "candidate_count": len(candidates),
                "selected_candidate": best.candidate.name,
                "selection_score": float(best.score),
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
    use_objective_strategy_defaults: bool = False,
    model_candidate_count: int = 1,
) -> dict[str, object]:
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
                source_scope="binance_full_history_public_market_data",
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
            write_status(
                "holdout_backtest_started",
                symbol_count_requested=len(selected),
                completed_symbol_count=len(metrics),
                current_symbol=item.symbol,
                current_symbol_index=selected_index,
                validation_rows=len(validation_rows),
            )
            result = run_backtest(
                validation_rows,
                model,
                evidence_strategy,
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
                    latency_ms=evidence_strategy.latency_buffer_ms,
                    liquidity_haircut=evidence_strategy.testnet_liquidity_haircut,
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
            baseline_series = _baseline_equity_series(validation_rows, starting_cash, evidence_strategy, market_type=market_type)
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
                risk_level=str(evidence_strategy.risk_level),
                leverage=float(evidence_strategy.leverage),
                effective_leverage=float(effective_leverage),
                leverage_applies=bool(leverage_applies),
                risk_per_trade=float(evidence_strategy.risk_per_trade),
                max_position_pct=float(evidence_strategy.max_position_pct),
                max_drawdown_limit_pct=float(evidence_strategy.max_drawdown_limit * 100.0),
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
        "health_filtered_symbol_selection": bool(not symbols and health_required),
        "selection_health_rejections": selection_health_rejections,
        "data_health_path": str(paths.data_health_path).replace("\\", "/"),
        "status_path": str(paths.status_path).replace("\\", "/"),
        "data_health": data_health,
        "selected_universe_path": str(paths.docs_data_dir / "selected-universe.json").replace("\\", "/"),
        "metrics_csv_path": str(paths.metrics_csv_path).replace("\\", "/"),
        "portfolio_timeline_csv_path": str(paths.timeline_csv_path).replace("\\", "/"),
        "progress_csv_path": str(paths.progress_csv_path).replace("\\", "/"),
        "progress": progress,
        "metrics": metric_rows,
        "tracked_artifacts": sorted(dict.fromkeys(tracked_artifacts)),
    }
    write_json_atomic(paths.report_path, report, indent=2, sort_keys=True)
    write_status(
        "round_complete",
        status="complete",
        symbol_count_requested=len(selected),
        completed_symbol_count=len(metrics),
        accepted_symbol_count=len(accepted_metrics),
        metrics_csv_path=str(paths.metrics_csv_path).replace("\\", "/"),
        report_path=str(paths.report_path).replace("\\", "/"),
    )
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
    "render_comparison_svg",
    "select_data_healthy_top_liquidity_symbols",
    "select_named_symbols",
    "select_top_liquidity_symbols",
    "strategy_with_objective_defaults",
]
