"""Reproducible optimization-round evidence and graph-data generation."""

from __future__ import annotations

import csv
import copy
import math
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .advanced_model import default_config_for, make_advanced_rows, train_advanced
from .api import BinanceAPIError, BinanceClient, Candle
from .assets import MAX_AUTONOMOUS_LEVERAGE
from .backtest import BacktestResult, calibrate_threshold_for_backtest, run_backtest
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
        progress_csv_path=docs_data_dir / "round-progress.csv",
        metrics_csv_path=docs_data_dir / "backtest-metrics.csv",
        timeline_csv_path=docs_data_dir / "portfolio-timeline.csv",
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
    if quality.coverage.count < min_rows:
        reasons.append(f"rows_below_min:{quality.coverage.count}/{min_rows}")
    if quality.gap_count > max_gap_count:
        reasons.append(f"gap_count_above_max:{quality.gap_count}/{max_gap_count}")
    if quality.coverage_ratio < min_coverage_ratio:
        reasons.append(f"coverage_ratio_below_min:{quality.coverage_ratio:.6f}/{min_coverage_ratio:.6f}")
    if archive_status_counts.get("error", 0) > 0:
        reasons.append(f"archive_errors:{archive_status_counts['error']}")
    if checksum_status_counts.get("mismatch", 0) > 0:
        reasons.append(f"checksum_mismatches:{checksum_status_counts['mismatch']}")
    if require_verified_checksum and checksum_status_counts.get("verified", 0) <= 0:
        reasons.append("no_verified_archive_checksum")
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
    }


def select_top_liquidity_symbols(
    client: BinanceClient,
    strategy: StrategyConfig,
    *,
    quote_asset: str = "USDT",
    count: int = 50,
    max_scan: int = 1000,
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
    ranked = [item for _score, item in sorted(candidates, key=lambda row: row[0], reverse=True)]
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


def _rolling_liquidity_flags(candles: Sequence[Candle], *, window: int = 96) -> dict[int, dict[str, float | int | bool | str]]:
    flags: dict[int, dict[str, float | int | bool | str]] = {}
    quote_volumes = [max(0.0, float(candle.quote_volume)) for candle in candles]
    trade_counts = [max(0, int(candle.trade_count)) for candle in candles]
    for index, candle in enumerate(candles):
        start = max(0, index - max(1, int(window)))
        window_volumes = quote_volumes[start:index]
        window_trades = trade_counts[start:index]
        volume = quote_volumes[index]
        trades = trade_counts[index]
        median_volume = statistics.median(window_volumes) if window_volumes else 0.0
        median_trades = statistics.median(window_trades) if window_trades else 0.0
        dt = datetime.fromtimestamp(int(candle.close_time) / 1000.0, tz=timezone.utc)
        bucket = _liquidity_clock_bucket(int(candle.close_time))
        bucket_volumes = [
            quote_volumes[prior]
            for prior in range(start, index)
            if _liquidity_clock_bucket(int(candles[prior].close_time)) == bucket
        ]
        bucket_trades = [
            trade_counts[prior]
            for prior in range(start, index)
            if _liquidity_clock_bucket(int(candles[prior].close_time)) == bucket
        ]
        bucket_median_volume = statistics.median(bucket_volumes) if len(bucket_volumes) >= 8 else 0.0
        bucket_median_trades = statistics.median(bucket_trades) if len(bucket_trades) >= 8 else 0.0
        low_volume = bool(median_volume > 0 and volume < median_volume * 0.35)
        low_trades = bool(median_trades > 0 and trades < median_trades * 0.35)
        low_bucket_volume = bool(bucket_median_volume > 0 and volume < bucket_median_volume * 0.45)
        low_bucket_trades = bool(bucket_median_trades > 0 and trades < bucket_median_trades * 0.45)
        flags[int(candle.close_time)] = {
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_comparison_svg(
    strategy_points: Sequence[EquityPoint],
    baseline_points: Sequence[EquityPoint],
    *,
    title: str,
    width: int = 1120,
    height: int = 660,
) -> str:
    points = list(strategy_points)
    baseline = list(baseline_points)
    if not points:
        points = [EquityPoint(0, 0.0, 0.0)]
    if not baseline:
        baseline = [EquityPoint(point.index, points[0].equity, 0.0, point.timestamp_ms) for point in points]
    left, right, top, bottom = 72, 34, 62, 82
    chart_w = width - left - right
    chart_h = height - top - bottom
    all_equity = [point.equity for point in points] + [point.equity for point in baseline]
    min_equity = min(all_equity)
    max_equity = max(all_equity)
    max_drawdown = max(0.01, *(point.drawdown for point in points))
    count = max(1, len(points) - 1)

    def sx(index: int) -> float:
        return left + (index / count) * chart_w

    def sy(value: float) -> float:
        if max_equity <= min_equity:
            return top + chart_h / 2.0
        return top + ((max_equity - value) / (max_equity - min_equity)) * chart_h

    def dy(value: float) -> float:
        return top + chart_h - (value / max_drawdown) * chart_h * 0.32

    strategy_poly = " ".join(f"{sx(i):.2f},{sy(point.equity):.2f}" for i, point in enumerate(points))
    baseline_poly = " ".join(f"{sx(i):.2f},{sy(point.equity):.2f}" for i, point in enumerate(baseline[:len(points)]))
    drawdown_poly = " ".join(f"{sx(i):.2f},{dy(point.drawdown):.2f}" for i, point in enumerate(points))
    timestamps = [point.timestamp_ms for point in points if point.timestamp_ms is not None]
    start = iso_utc(min(timestamps))[:10] if timestamps else "sample"
    end = iso_utc(max(timestamps))[:10] if timestamps else "index"
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
  <text x="{left}" y="{height - 28}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">UTC span: {start} to {end}. Chart source: committed CSV graph data.</text>
  <text x="{left}" y="{top + chart_h + 18}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">{start}</text>
  <text x="{left + chart_w - 84}" y="{top + chart_h + 18}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">{end}</text>
</svg>
"""


def _portfolio_timeline(rows_by_symbol: Sequence[Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    by_timestamp: dict[int, list[Mapping[str, object]]] = {}
    for rows in rows_by_symbol:
        for row in rows:
            try:
                timestamp = int(row.get("timestamp_ms", 0))
            except (TypeError, ValueError):
                continue
            if timestamp <= 0:
                continue
            by_timestamp.setdefault(timestamp, []).append(row)
    output: list[dict[str, object]] = []
    for timestamp in sorted(by_timestamp):
        items = by_timestamp[timestamp]
        strategy_values = [_finite(item.get("strategy_equity")) for item in items]
        baseline_values = [_finite(item.get("baseline_equity")) for item in items]
        drawdowns = [_finite(item.get("strategy_drawdown")) for item in items]
        low_liquidity = sum(1 for item in items if str(item.get("low_liquidity_flag")).lower() == "true")
        output.append({
            "timestamp_ms": timestamp,
            "timestamp_utc": iso_utc(timestamp),
            "symbols_reporting": len(items),
            "mean_strategy_equity": sum(strategy_values) / len(strategy_values) if strategy_values else 0.0,
            "mean_baseline_equity": sum(baseline_values) / len(baseline_values) if baseline_values else 0.0,
            "mean_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0.0,
            "low_liquidity_symbol_count": low_liquidity,
        })
    return output


def train_round_model(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str,
    batch_size: int,
) -> tuple[object, object, list[object], list[object]]:
    feature_cfg = default_config_for(objective.name, strategy.enabled_features)
    rows = make_advanced_rows(candles, feature_cfg)
    train_selection_rows, validation_rows = _split_train_validation(rows, validation_fraction=0.25)
    train_rows, selection_rows = _split_train_validation(train_selection_rows, validation_fraction=0.20)
    if not train_rows or not selection_rows or not validation_rows:
        raise ValueError("insufficient rows for train/validation backtest evidence")
    model, report = train_advanced(
        train_rows,
        feature_cfg,
        epochs=max(1, int(objective.training.epochs if objective.training else 100)),
        learning_rate=float(objective.training.learning_rate if objective.training else 0.03),
        l2_penalty=float(objective.training.l2_penalty if objective.training else 1e-3),
        validation_rows=selection_rows[: max(1, min(len(selection_rows), 5000))],
        early_stopping_rounds=30,
        compute_backend=compute_backend,
        batch_size=batch_size,
    )
    calibration = calibrate_probability_temperature(list(selection_rows[: max(1, min(len(selection_rows), 5000))]), model)
    if calibration.status != "fail":
        model.probability_temperature = float(calibration.temperature)
        model.probability_calibration_size = int(calibration.rows)
    model.decision_threshold = float(objective.training.signal_threshold if objective.training else strategy.signal_threshold)
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
        steps=19,
        min_score_delta=0.0,
        compute_backend=compute_backend,
        score_batch_size=batch_size,
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
    base_score = objective.score(base_result) if objective.accepts(base_result) else float("-inf")
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
    inverted_score = objective.score(inverted_result) if objective.accepts(inverted_result) else float("-inf")
    if inverted_score > base_score + 1e-12:
        model = inverted_model
    elif not math.isfinite(base_score):
        model.meta_label_policy = {
            "enabled": True,
            "mode": "take_downsize_skip",
            "reason": "round_selection_gate_failed",
            "objective": objective.name,
            "target_precision": 1.0,
            "take_threshold": 1_000_000_000.0,
            "downsize_threshold": 1_000_000_000.0,
            "downsize_fraction": 0.05,
            "sample_count": 0,
        }
        model.threshold_source = "round_selection_fail_closed"
        model.quality_warnings = [
            *list(getattr(model, "quality_warnings", [])),
            "round_selection_gate_failed_no_final_holdout_entries",
        ]
    return model, report, list(rows), list(validation_rows)


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
    use_objective_strategy_defaults: bool = False,
) -> dict[str, object]:
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
    write_json_atomic(paths.docs_data_dir / "selected-universe.json", [item.asdict() for item in selected], indent=2, sort_keys=True)
    metrics: list[BacktestEvidence] = []
    data_health: list[dict[str, object]] = []
    timeline_rows_all: list[list[dict[str, object]]] = []
    for item in selected:
        candles: list[Candle] = []
        coverage = None
        try:
            if item.tier == "explicit-symbol" and item.reasons:
                reasons = ", ".join(item.reasons)
                raise ValueError(f"symbol_selection_failed: {reasons}")
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
            if health_required and health.get("status") != "ok":
                reasons = ", ".join(str(reason) for reason in health.get("reasons", []) if reason)
                raise ValueError(f"data_health_failed: {reasons or 'unknown'}")
            candles = fetch_full_history(
                client,
                item.symbol,
                interval,
                db_path=paths.market_db_path,
                market_type=market_type,
                batch_size=batch_size,
                allow_network_backfill=not require_prefilled_data,
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
            model, train_report, rows, validation_rows = train_round_model(
                candles,
                evidence_strategy,
                objective,
                market_type=market_type,
                starting_cash=starting_cash,
                compute_backend=compute_backend,
                batch_size=batch_size,
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
            market_edge = build_market_edge_report(result, objective)
            accepted = bool(objective.accepts(result) and market_edge.accepted and not result.stopped_by_drawdown)
            reason = objective.reject_reason(result) or market_edge.reason
            strategy_points = _result_points(result)
            baseline_series = _baseline_equity_series(validation_rows, starting_cash, evidence_strategy, market_type=market_type)
            baseline_points = [
                EquityPoint(index=index, equity=_finite(point.get("equity")), drawdown=_finite(point.get("drawdown")), timestamp_ms=int(point["timestamp"]))
                for index, point in enumerate(baseline_series)
                if "timestamp" in point
            ]
            liquidity_by_close = _rolling_liquidity_flags(candles)
            timeline_rows: list[dict[str, object]] = []
            point_by_timestamp = {int(point.timestamp_ms or 0): point for point in strategy_points}
            baseline_by_timestamp = {int(point.timestamp_ms or 0): point for point in baseline_points}
            for timestamp, point in sorted(point_by_timestamp.items()):
                if timestamp <= 0:
                    continue
                baseline = baseline_by_timestamp.get(timestamp)
                liquidity = liquidity_by_close.get(timestamp, {})
                timeline_rows.append({
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
                })
            symbol_timeline_path = paths.docs_data_dir / f"{item.rank:02d}-{item.symbol}-{objective.name}-timeline.csv"
            _write_csv(
                symbol_timeline_path,
                timeline_rows,
                (
                    "round_id", "symbol", "objective", "timestamp_ms", "timestamp_utc",
                    "strategy_equity", "baseline_equity", "strategy_drawdown", "baseline_drawdown",
                    "quote_volume", "trade_count", "rolling_quote_volume_median",
                    "rolling_trade_count_median", "clock_bucket", "clock_bucket_quote_volume_median",
                    "clock_bucket_trade_count_median", "data_probed_low_session_flag",
                    "low_liquidity_flag", "weekend_flag", "utc_hour", "utc_weekday",
                ),
            )
            timeline_rows_all.append(timeline_rows)
            chart_path = paths.docs_charts_dir / f"{item.rank:02d}-{item.symbol}-{objective.name}.svg"
            chart_path.write_text(
                render_comparison_svg(strategy_points, baseline_points, title=f"{item.symbol} {objective.label} Backtest vs Passive Baseline"),
                encoding="utf-8",
            )
            low_rate = (
                sum(1 for row in timeline_rows if str(row.get("low_liquidity_flag")).lower() == "true") / len(timeline_rows)
                if timeline_rows else 0.0
            )
            weekend_rate = (
                sum(1 for row in timeline_rows if str(row.get("weekend_flag")).lower() == "true") / len(timeline_rows)
                if timeline_rows else 0.0
            )
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
    metric_rows = [metric.asdict() for metric in metrics]
    _write_csv(
        paths.metrics_csv_path,
        metric_rows,
        tuple(metric_rows[0].keys()) if metric_rows else ("round_id", "symbol", "objective"),
    )
    portfolio_rows = _portfolio_timeline(timeline_rows_all)
    _write_csv(
        paths.timeline_csv_path,
        portfolio_rows,
        (
            "timestamp_ms", "timestamp_utc", "symbols_reporting", "mean_strategy_equity",
            "mean_baseline_equity", "mean_drawdown", "low_liquidity_symbol_count",
        ),
    )
    accepted_metrics = [metric for metric in metrics if metric.accepted]
    tracked_artifacts = [
        str(paths.progress_csv_path).replace("\\", "/"),
        str(paths.metrics_csv_path).replace("\\", "/"),
        str(paths.timeline_csv_path).replace("\\", "/"),
        str(paths.report_path).replace("\\", "/"),
        str(paths.data_health_path).replace("\\", "/"),
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
        "mean_low_liquidity_sample_rate_pct": statistics.mean([metric.low_liquidity_sample_rate_pct for metric in metrics]) if metrics else 0.0,
    }
    _write_csv(paths.progress_csv_path, [progress], tuple(progress.keys()))
    write_json_atomic(paths.data_health_path, data_health, indent=2, sort_keys=True)
    report = {
        "round_id": round_id,
        "generated_at_utc": _utc_now(),
        "artifact_class": "exchange_sourced_backtest_graph_data",
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
    return report


__all__ = [
    "BacktestEvidence",
    "EvidencePaths",
    "SelectedSymbol",
    "build_round_evidence",
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
