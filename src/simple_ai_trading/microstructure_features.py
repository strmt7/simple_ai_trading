"""Causal L1/tape features and executable after-cost labels from the tick warehouse."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import numpy as np
from numba import njit, prange

from .assets import normalize_symbol
from .microstructure_warehouse import MicrostructureWarehouse


MICROSTRUCTURE_FEATURE_VERSION = "l1-tape-causal-v7"
MICROSTRUCTURE_TRADE_EMBARGO_MS = 1_000


@dataclass(frozen=True)
class MicrostructureDataset:
    symbol: str
    feature_version: str
    feature_names: tuple[str, ...]
    horizon_seconds: int
    total_latency_ms: int
    taker_fee_bps: float
    additional_slippage_bps_per_side: float
    reference_order_notional_quote: float
    max_l1_participation: float
    max_quote_age_ms: int
    decision_cadence_seconds: int
    target_mode: str
    stop_loss_bps: float | None
    take_profit_bps: float | None
    trigger_execution_slippage_bps: float | None
    path_resolution_ms: int | None
    decision_time_ms: np.ndarray
    long_exit_time_ms: np.ndarray
    short_exit_time_ms: np.ndarray
    features: np.ndarray
    long_net_bps: np.ndarray
    short_net_bps: np.ndarray
    entry_spread_bps: np.ndarray
    exit_spread_bps: np.ndarray
    entry_quote_age_ms: np.ndarray
    exit_quote_age_ms: np.ndarray
    entry_bid_price: np.ndarray
    entry_ask_price: np.ndarray
    fixed_exit_bid_price: np.ndarray
    fixed_exit_ask_price: np.ndarray
    entry_bid_qty: np.ndarray
    entry_ask_qty: np.ndarray
    fixed_exit_bid_qty: np.ndarray
    fixed_exit_ask_qty: np.ndarray
    long_l1_participation: np.ndarray
    short_l1_participation: np.ndarray
    long_liquidity_eligible: np.ndarray
    short_liquidity_eligible: np.ndarray
    source_evidence: Mapping[str, object] | None = None
    trade_feature_embargo_ms: int = MICROSTRUCTURE_TRADE_EMBARGO_MS

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def best_side_net_bps(self) -> np.ndarray:
        long = np.where(self.long_liquidity_eligible, self.long_net_bps, -np.inf)
        short = np.where(self.short_liquidity_eligible, self.short_net_bps, -np.inf)
        output = np.maximum(long, short)
        return np.where(np.isfinite(output), output, np.nan)

    @property
    def best_side(self) -> np.ndarray:
        long = np.where(self.long_liquidity_eligible, self.long_net_bps, -np.inf)
        short = np.where(self.short_liquidity_eligible, self.short_net_bps, -np.inf)
        return np.where(
            ~np.isfinite(np.maximum(long, short)),
            0,
            np.where(long >= short, 1, -1),
        ).astype(np.int8)

    def summary(self) -> dict[str, object]:
        best = self.best_side_net_bps
        executable_best = best[np.isfinite(best)]
        long_executable = self.long_net_bps[self.long_liquidity_eligible]
        short_executable = self.short_net_bps[self.short_liquidity_eligible]
        return {
            "symbol": self.symbol,
            "feature_version": self.feature_version,
            "rows": self.rows,
            "feature_count": len(self.feature_names),
            "first_decision_time_ms": int(self.decision_time_ms[0]) if self.rows else None,
            "last_decision_time_ms": int(self.decision_time_ms[-1]) if self.rows else None,
            "horizon_seconds": self.horizon_seconds,
            "total_latency_ms": self.total_latency_ms,
            "taker_fee_bps": self.taker_fee_bps,
            "additional_slippage_bps_per_side": (
                self.additional_slippage_bps_per_side
            ),
            "reference_order_notional_quote": self.reference_order_notional_quote,
            "max_l1_participation": self.max_l1_participation,
            "max_quote_age_ms": self.max_quote_age_ms,
            "decision_cadence_seconds": self.decision_cadence_seconds,
            "target_mode": self.target_mode,
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "trigger_execution_slippage_bps": self.trigger_execution_slippage_bps,
            "path_resolution_ms": self.path_resolution_ms,
            "trade_feature_embargo_ms": self.trade_feature_embargo_ms,
            "mean_long_holding_seconds": (
                float(np.mean(self.long_exit_time_ms - self.decision_time_ms) / 1000.0)
                if self.rows
                else None
            ),
            "mean_short_holding_seconds": (
                float(np.mean(self.short_exit_time_ms - self.decision_time_ms) / 1000.0)
                if self.rows
                else None
            ),
            "mean_long_net_bps": float(np.mean(self.long_net_bps)) if self.rows else None,
            "mean_short_net_bps": float(np.mean(self.short_net_bps)) if self.rows else None,
            "mean_oracle_best_side_net_bps": (
                float(np.mean(executable_best)) if executable_best.size else None
            ),
            "positive_long_ratio": (
                float(np.mean(long_executable > 0.0)) if long_executable.size else None
            ),
            "positive_short_ratio": (
                float(np.mean(short_executable > 0.0)) if short_executable.size else None
            ),
            "positive_oracle_ratio": (
                float(np.mean(executable_best > 0.0)) if executable_best.size else None
            ),
            "long_liquidity_eligible_ratio": (
                float(np.mean(self.long_liquidity_eligible)) if self.rows else None
            ),
            "short_liquidity_eligible_ratio": (
                float(np.mean(self.short_liquidity_eligible)) if self.rows else None
            ),
            "long_l1_participation_p99": (
                float(np.quantile(self.long_l1_participation, 0.99)) if self.rows else None
            ),
            "short_l1_participation_p99": (
                float(np.quantile(self.short_l1_participation, 0.99)) if self.rows else None
            ),
            "entry_quote_age_p99_ms": float(np.quantile(self.entry_quote_age_ms, 0.99)) if self.rows else None,
            "exit_quote_age_p99_ms": float(np.quantile(self.exit_quote_age_ms, 0.99)) if self.rows else None,
            "source_evidence": dict(self.source_evidence) if self.source_evidence is not None else None,
        }


_FEATURE_COLUMNS = (
    "return_1s_bps",
    "return_5s_bps",
    "return_15s_bps",
    "return_30s_bps",
    "return_60s_bps",
    "return_120s_bps",
    "return_300s_bps",
    "return_900s_bps",
    "realized_volatility_10s_bps",
    "realized_volatility_30s_bps",
    "realized_volatility_60s_bps",
    "realized_volatility_120s_bps",
    "realized_volatility_300s_bps",
    "realized_volatility_900s_bps",
    "intrasecond_range_bps",
    "range_60s_bps",
    "range_300s_bps",
    "range_900s_bps",
    "spread_bps",
    "max_spread_bps",
    "spread_vs_60s_mean",
    "spread_vs_300s_mean",
    "l1_imbalance",
    "close_l1_imbalance",
    "imbalance_10s_mean",
    "imbalance_60s_mean",
    "imbalance_300s_mean",
    "microprice_offset_bps",
    "normalized_ofi",
    "ofi_10s_mean",
    "ofi_60s_mean",
    "ofi_300s_mean",
    "ofi_delta_5s",
    "ofi_delta_15s",
    "ofi_delta_30s",
    "ofi_delta_60s",
    "log_quote_updates",
    "quote_intensity_vs_60s_mean",
    "quote_intensity_vs_300s_mean",
    "trade_imbalance",
    "trade_imbalance_10s_mean",
    "trade_imbalance_60s_mean",
    "trade_imbalance_300s_mean",
    "trade_imbalance_delta_5s",
    "trade_imbalance_delta_15s",
    "trade_imbalance_delta_30s",
    "trade_imbalance_delta_60s",
    "signed_flow_10s",
    "signed_flow_60s",
    "signed_flow_300s",
    "log_base_volume",
    "volume_vs_60s_mean",
    "volume_vs_300s_mean",
    "log_trade_count",
    "trade_close_vs_mid_bps",
    "event_delay_p50_ms",
    "event_delay_p99_ms",
    "event_delay_vs_60s_mean",
    "return_efficiency_60s",
    "return_efficiency_300s",
    "l1_imbalance_delta_5s",
    "l1_imbalance_delta_15s",
    "l1_imbalance_delta_30s",
    "l1_imbalance_delta_60s",
    "microprice_delta_5s_bps",
    "microprice_delta_15s_bps",
    "microprice_delta_30s_bps",
    "microprice_delta_60s_bps",
    "return_60s_vol_units",
    "return_300s_vol_units",
    "return_900s_vol_units",
    "volatility_10s_vs_300s",
    "volatility_60s_vs_900s",
    "spread_to_10s_volatility",
    "ofi_trade_flow_agreement",
    "quote_trade_log_intensity_gap",
    "intrasecond_close_location",
    "utc_day_sin",
    "utc_day_cos",
    "funding_cycle_sin",
    "funding_cycle_cos",
    "return_1800s_bps",
    "return_3600s_bps",
    "realized_volatility_1800s_bps",
    "realized_volatility_3600s_bps",
    "range_1800s_bps",
    "range_3600s_bps",
    "spread_vs_900s_mean",
    "quote_intensity_vs_900s_mean",
    "volume_vs_900s_mean",
    "trade_intensity_vs_900s_mean",
    "return_efficiency_900s",
    "return_efficiency_3600s",
    "return_1800s_vol_units",
    "return_3600s_vol_units",
    "volatility_300s_vs_3600s",
    "volatility_900s_vs_3600s",
    "utc_week_sin",
    "utc_week_cos",
    "weekend_flag",
)
MICROSTRUCTURE_FEATURE_NAMES = _FEATURE_COLUMNS


def build_executable_microstructure_dataset(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    horizon_seconds: int,
    total_latency_ms: int,
    taker_fee_bps: float,
    additional_slippage_bps_per_side: float = 0.0,
    max_quote_age_ms: int = 1_000,
    reference_order_notional_quote: float = 1_000.0,
    max_l1_participation: float = 0.10,
    decision_cadence_seconds: int = 5,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> MicrostructureDataset:
    """Build strictly causal features with real bid/ask entry and exit labels."""

    normalized_symbol = normalize_symbol(symbol)
    horizon = int(horizon_seconds)
    latency = int(total_latency_ms)
    fee = float(taker_fee_bps)
    additional_slippage = float(additional_slippage_bps_per_side)
    max_age = int(max_quote_age_ms)
    reference_notional = float(reference_order_notional_quote)
    participation_limit = float(max_l1_participation)
    decision_cadence = int(decision_cadence_seconds)
    if horizon <= 0:
        raise ValueError("horizon_seconds must be positive")
    if latency < 0:
        raise ValueError("total_latency_ms must be non-negative")
    if not math.isfinite(fee) or fee < 0.0:
        raise ValueError("taker_fee_bps must be finite and non-negative")
    if not math.isfinite(additional_slippage) or additional_slippage < 0.0:
        raise ValueError(
            "additional_slippage_bps_per_side must be finite and non-negative"
        )
    if max_age <= 0:
        raise ValueError("max_quote_age_ms must be positive")
    if not math.isfinite(reference_notional) or reference_notional <= 0.0:
        raise ValueError("reference_order_notional_quote must be finite and positive")
    if not math.isfinite(participation_limit) or not 0.0 < participation_limit <= 1.0:
        raise ValueError("max_l1_participation must lie in (0, 1]")
    if decision_cadence <= 0 or decision_cadence > 60:
        raise ValueError("decision_cadence_seconds must lie in [1, 60]")

    lower = -9_223_372_036_854_775_808 if start_ms is None else int(start_ms)
    upper = 9_223_372_036_854_775_807 if end_ms is None else int(end_ms)
    if lower > upper:
        raise ValueError("start_ms must be less than or equal to end_ms")

    source_evidence = None
    require_causal_bars = getattr(warehouse, "require_causal_feature_bars", None)
    if callable(require_causal_bars):
        source_evidence = require_causal_bars(normalized_symbol)
        require_corpus = getattr(warehouse, "require_corpus_certificate", None)
        if callable(require_corpus):
            certificate_kwargs: dict[str, object] = {
                "required_data_types": ("bookTicker", "trades"),
                "require_full_history_inventory": True,
            }
            if start_ms is not None and end_ms is not None:
                certificate_kwargs.update(
                    {
                        "required_start_ms": int(start_ms),
                        "required_end_ms": int(end_ms),
                    }
                )
            corpus_certificate = require_corpus(
                normalized_symbol,
                **certificate_kwargs,
            )
            source_evidence = {
                **dict(source_evidence),
                "corpus_certificate": corpus_certificate,
            }

    sql = """
        WITH joined AS (
            SELECT
                q.symbol, q.second_ms, q.open_mid, q.high_mid, q.low_mid, q.close_mid,
                q.close_bid, q.close_ask, q.close_bid_qty, q.close_ask_qty,
                q.event_weighted_spread_bps AS spread_bps,
                q.max_spread_bps,
                q.event_weighted_l1_imbalance AS l1_imbalance,
                q.close_l1_imbalance,
                q.event_weighted_microprice_offset_bps AS microprice_offset_bps,
                q.quote_updates,
                q.event_delay_p50_ms, q.event_delay_p99_ms,
                coalesce(t.close, q.close_mid) AS trade_close,
                coalesce(t.base_volume, 0.0) AS base_volume,
                coalesce(t.quote_volume, 0.0) AS quote_volume,
                coalesce(t.aggressive_buy_volume, 0.0) AS aggressive_buy_volume,
                coalesce(t.aggressive_sell_volume, 0.0) AS aggressive_sell_volume,
                coalesce(t.trade_imbalance, 0.0) AS trade_imbalance,
                coalesce(t.trade_count, 0) AS trade_count
            FROM current_book_ticker_1s q
            LEFT JOIN current_trade_1s t
              ON q.symbol = t.symbol
             AND t.second_ms + ? = q.second_ms
            WHERE q.symbol = ? AND q.second_ms BETWEEN ? AND ?
        ),
        lagged AS (
            SELECT *,
                lag(second_ms) OVER sequence AS previous_second_ms,
                lag(close_mid) OVER sequence AS previous_mid,
                lag(close_bid) OVER sequence AS previous_bid,
                lag(close_ask) OVER sequence AS previous_ask,
                lag(close_bid_qty) OVER sequence AS previous_bid_qty,
                lag(close_ask_qty) OVER sequence AS previous_ask_qty
            FROM joined
            WINDOW sequence AS (ORDER BY second_ms)
        ),
        instantaneous AS (
            SELECT *,
                CASE WHEN previous_second_ms = second_ms - 1000
                     THEN ln(close_mid / previous_mid) END AS log_return_1s,
                (
                    CASE WHEN close_bid >= previous_bid THEN close_bid_qty ELSE 0.0 END
                    - CASE WHEN close_bid <= previous_bid THEN previous_bid_qty ELSE 0.0 END
                    - CASE WHEN close_ask <= previous_ask THEN close_ask_qty ELSE 0.0 END
                    + CASE WHEN close_ask >= previous_ask THEN previous_ask_qty ELSE 0.0 END
                ) / greatest(
                    1e-12,
                    (close_bid_qty + close_ask_qty + previous_bid_qty + previous_ask_qty) / 2.0
                ) AS normalized_ofi
            FROM lagged
        ),
        stationary AS (
            SELECT *,
                lag(log_return_1s) OVER (ORDER BY second_ms) AS previous_log_return_1s,
                aggressive_buy_volume - aggressive_sell_volume AS signed_base_flow
            FROM instantaneous
        ),
        rolling AS (
            SELECT *,
                sum(log_return_1s) OVER range_5s AS log_return_5s,
                sum(log_return_1s) OVER range_15s AS log_return_15s,
                sum(log_return_1s) OVER range_30s AS log_return_30s,
                sum(log_return_1s) OVER range_60s AS log_return_60s,
                sum(log_return_1s) OVER range_120s AS log_return_120s,
                sum(log_return_1s) OVER range_300s AS log_return_300s,
                sum(log_return_1s) OVER range_900s AS log_return_900s,
                sum(log_return_1s) OVER range_1800s AS log_return_1800s,
                sum(log_return_1s) OVER range_3600s AS log_return_3600s,
                stddev_pop(log_return_1s) OVER range_10s AS volatility_10s,
                stddev_pop(log_return_1s) OVER range_30s AS volatility_30s,
                stddev_pop(log_return_1s) OVER range_60s AS volatility_60s,
                stddev_pop(log_return_1s) OVER range_120s AS volatility_120s,
                stddev_pop(log_return_1s) OVER range_300s AS volatility_300s,
                stddev_pop(log_return_1s) OVER range_900s AS volatility_900s,
                stddev_pop(log_return_1s) OVER range_1800s AS volatility_1800s,
                stddev_pop(log_return_1s) OVER range_3600s AS volatility_3600s,
                (max(high_mid) OVER range_60s - min(low_mid) OVER range_60s)
                    * 10000.0 / close_mid AS range_60s_bps,
                (max(high_mid) OVER range_300s - min(low_mid) OVER range_300s)
                    * 10000.0 / close_mid AS range_300s_bps,
                (max(high_mid) OVER range_900s - min(low_mid) OVER range_900s)
                    * 10000.0 / close_mid AS range_900s_bps,
                (max(high_mid) OVER range_1800s - min(low_mid) OVER range_1800s)
                    * 10000.0 / close_mid AS range_1800s_bps,
                (max(high_mid) OVER range_3600s - min(low_mid) OVER range_3600s)
                    * 10000.0 / close_mid AS range_3600s_bps,
                avg(spread_bps) OVER range_60s AS spread_60s_mean,
                avg(spread_bps) OVER range_300s AS spread_300s_mean,
                avg(spread_bps) OVER range_900s AS spread_900s_mean,
                avg(l1_imbalance) OVER range_10s AS imbalance_10s_mean,
                avg(l1_imbalance) OVER range_60s AS imbalance_60s_mean,
                avg(l1_imbalance) OVER range_300s AS imbalance_300s_mean,
                avg(normalized_ofi) OVER range_10s AS ofi_10s_mean,
                avg(normalized_ofi) OVER range_60s AS ofi_60s_mean,
                avg(normalized_ofi) OVER range_300s AS ofi_300s_mean,
                avg(quote_updates) OVER range_60s AS quote_updates_60s_mean,
                avg(quote_updates) OVER range_300s AS quote_updates_300s_mean,
                avg(quote_updates) OVER range_900s AS quote_updates_900s_mean,
                avg(trade_imbalance) OVER range_10s AS trade_imbalance_10s_mean,
                avg(trade_imbalance) OVER range_60s AS trade_imbalance_60s_mean,
                avg(trade_imbalance) OVER range_300s AS trade_imbalance_300s_mean,
                sum(signed_base_flow) OVER range_10s
                    / greatest(sum(base_volume) OVER range_10s, 1e-12) AS signed_flow_10s,
                sum(signed_base_flow) OVER range_60s
                    / greatest(sum(base_volume) OVER range_60s, 1e-12) AS signed_flow_60s,
                sum(signed_base_flow) OVER range_300s
                    / greatest(sum(base_volume) OVER range_300s, 1e-12) AS signed_flow_300s,
                avg(base_volume) OVER range_60s AS base_volume_60s_mean,
                avg(base_volume) OVER range_300s AS base_volume_300s_mean,
                avg(base_volume) OVER range_900s AS base_volume_900s_mean,
                avg(trade_count) OVER range_900s AS trade_count_900s_mean,
                avg(event_delay_p99_ms) OVER range_60s AS event_delay_60s_mean,
                abs(sum(log_return_1s) OVER range_60s)
                    / greatest(sum(abs(log_return_1s)) OVER range_60s, 1e-12) AS return_efficiency_60s,
                abs(sum(log_return_1s) OVER range_300s)
                    / greatest(sum(abs(log_return_1s)) OVER range_300s, 1e-12) AS return_efficiency_300s,
                abs(sum(log_return_1s) OVER range_900s)
                    / greatest(sum(abs(log_return_1s)) OVER range_900s, 1e-12) AS return_efficiency_900s,
                abs(sum(log_return_1s) OVER range_3600s)
                    / greatest(sum(abs(log_return_1s)) OVER range_3600s, 1e-12) AS return_efficiency_3600s,
                count(*) OVER range_60s AS observations_60s,
                count(*) OVER range_900s AS observations_900s,
                count(*) OVER range_3600s AS observations_3600s
            FROM stationary
            WINDOW
                range_5s AS (ORDER BY second_ms RANGE BETWEEN 4000 PRECEDING AND CURRENT ROW),
                range_10s AS (ORDER BY second_ms RANGE BETWEEN 9000 PRECEDING AND CURRENT ROW),
                range_15s AS (ORDER BY second_ms RANGE BETWEEN 14000 PRECEDING AND CURRENT ROW),
                range_30s AS (ORDER BY second_ms RANGE BETWEEN 29000 PRECEDING AND CURRENT ROW),
                range_60s AS (ORDER BY second_ms RANGE BETWEEN 59000 PRECEDING AND CURRENT ROW),
                range_120s AS (ORDER BY second_ms RANGE BETWEEN 119000 PRECEDING AND CURRENT ROW),
                 range_300s AS (ORDER BY second_ms RANGE BETWEEN 299000 PRECEDING AND CURRENT ROW),
                 range_900s AS (ORDER BY second_ms RANGE BETWEEN 899000 PRECEDING AND CURRENT ROW),
                 range_1800s AS (ORDER BY second_ms RANGE BETWEEN 1799000 PRECEDING AND CURRENT ROW),
                 range_3600s AS (ORDER BY second_ms RANGE BETWEEN 3599000 PRECEDING AND CURRENT ROW)
        ),
        temporal AS (
            SELECT *,
                lag(second_ms, 60) OVER sequence AS lag_60_second_ms,
                lag(normalized_ofi, 5) OVER sequence AS ofi_lag_5s,
                lag(normalized_ofi, 15) OVER sequence AS ofi_lag_15s,
                lag(normalized_ofi, 30) OVER sequence AS ofi_lag_30s,
                lag(normalized_ofi, 60) OVER sequence AS ofi_lag_60s,
                lag(trade_imbalance, 5) OVER sequence AS trade_imbalance_lag_5s,
                lag(trade_imbalance, 15) OVER sequence AS trade_imbalance_lag_15s,
                lag(trade_imbalance, 30) OVER sequence AS trade_imbalance_lag_30s,
                lag(trade_imbalance, 60) OVER sequence AS trade_imbalance_lag_60s,
                lag(close_l1_imbalance, 5) OVER sequence AS l1_imbalance_lag_5s,
                lag(close_l1_imbalance, 15) OVER sequence AS l1_imbalance_lag_15s,
                lag(close_l1_imbalance, 30) OVER sequence AS l1_imbalance_lag_30s,
                lag(close_l1_imbalance, 60) OVER sequence AS l1_imbalance_lag_60s,
                lag(microprice_offset_bps, 5) OVER sequence AS microprice_lag_5s,
                lag(microprice_offset_bps, 15) OVER sequence AS microprice_lag_15s,
                lag(microprice_offset_bps, 30) OVER sequence AS microprice_lag_30s,
                lag(microprice_offset_bps, 60) OVER sequence AS microprice_lag_60s
            FROM rolling
            WINDOW sequence AS (ORDER BY second_ms)
        ),
        decisions AS (
            SELECT
                *,
                second_ms + 1000 AS decision_time_ms,
                second_ms + 1000 + ? AS entry_arrival_ms,
                second_ms + 1000 + ? + ? * 1000 AS exit_arrival_ms,
                log_return_1s * 10000.0 AS return_1s_bps,
                log_return_5s * 10000.0 AS return_5s_bps,
                log_return_15s * 10000.0 AS return_15s_bps,
                log_return_30s * 10000.0 AS return_30s_bps,
                log_return_60s * 10000.0 AS return_60s_bps,
                log_return_120s * 10000.0 AS return_120s_bps,
                log_return_300s * 10000.0 AS return_300s_bps,
                log_return_900s * 10000.0 AS return_900s_bps,
                log_return_1800s * 10000.0 AS return_1800s_bps,
                log_return_3600s * 10000.0 AS return_3600s_bps,
                volatility_10s * 10000.0 AS realized_volatility_10s_bps,
                volatility_30s * 10000.0 AS realized_volatility_30s_bps,
                volatility_60s * 10000.0 AS realized_volatility_60s_bps,
                volatility_120s * 10000.0 AS realized_volatility_120s_bps,
                volatility_300s * 10000.0 AS realized_volatility_300s_bps,
                volatility_900s * 10000.0 AS realized_volatility_900s_bps,
                volatility_1800s * 10000.0 AS realized_volatility_1800s_bps,
                volatility_3600s * 10000.0 AS realized_volatility_3600s_bps,
                (high_mid - low_mid) * 10000.0 / close_mid AS intrasecond_range_bps,
                spread_bps / greatest(spread_60s_mean, 1e-12) AS spread_vs_60s_mean,
                spread_bps / greatest(spread_300s_mean, 1e-12) AS spread_vs_300s_mean,
                spread_bps / greatest(spread_900s_mean, 1e-12) AS spread_vs_900s_mean,
                ln(1.0 + quote_updates) AS log_quote_updates,
                quote_updates / greatest(quote_updates_60s_mean, 1e-12) AS quote_intensity_vs_60s_mean,
                quote_updates / greatest(quote_updates_300s_mean, 1e-12) AS quote_intensity_vs_300s_mean,
                quote_updates / greatest(quote_updates_900s_mean, 1e-12) AS quote_intensity_vs_900s_mean,
                ln(1.0 + base_volume) AS log_base_volume,
                base_volume / greatest(base_volume_60s_mean, 1e-12) AS volume_vs_60s_mean,
                base_volume / greatest(base_volume_300s_mean, 1e-12) AS volume_vs_300s_mean,
                base_volume / greatest(base_volume_900s_mean, 1e-12) AS volume_vs_900s_mean,
                ln(1.0 + trade_count) AS log_trade_count,
                trade_count / greatest(trade_count_900s_mean, 1e-12) AS trade_intensity_vs_900s_mean,
                (trade_close / close_mid - 1.0) * 10000.0 AS trade_close_vs_mid_bps,
                event_delay_p99_ms / greatest(event_delay_60s_mean, 1e-12) AS event_delay_vs_60s_mean,
                normalized_ofi - ofi_lag_5s AS ofi_delta_5s,
                normalized_ofi - ofi_lag_15s AS ofi_delta_15s,
                normalized_ofi - ofi_lag_30s AS ofi_delta_30s,
                normalized_ofi - ofi_lag_60s AS ofi_delta_60s,
                trade_imbalance - trade_imbalance_lag_5s AS trade_imbalance_delta_5s,
                trade_imbalance - trade_imbalance_lag_15s AS trade_imbalance_delta_15s,
                trade_imbalance - trade_imbalance_lag_30s AS trade_imbalance_delta_30s,
                trade_imbalance - trade_imbalance_lag_60s AS trade_imbalance_delta_60s,
                close_l1_imbalance - l1_imbalance_lag_5s AS l1_imbalance_delta_5s,
                close_l1_imbalance - l1_imbalance_lag_15s AS l1_imbalance_delta_15s,
                close_l1_imbalance - l1_imbalance_lag_30s AS l1_imbalance_delta_30s,
                close_l1_imbalance - l1_imbalance_lag_60s AS l1_imbalance_delta_60s,
                microprice_offset_bps - microprice_lag_5s AS microprice_delta_5s_bps,
                microprice_offset_bps - microprice_lag_15s AS microprice_delta_15s_bps,
                microprice_offset_bps - microprice_lag_30s AS microprice_delta_30s_bps,
                microprice_offset_bps - microprice_lag_60s AS microprice_delta_60s_bps,
                log_return_60s / greatest(volatility_60s * sqrt(60.0), 1e-12)
                    AS return_60s_vol_units,
                log_return_300s / greatest(volatility_300s * sqrt(300.0), 1e-12)
                    AS return_300s_vol_units,
                log_return_900s / greatest(volatility_900s * sqrt(900.0), 1e-12)
                    AS return_900s_vol_units,
                log_return_1800s / greatest(volatility_1800s * sqrt(1800.0), 1e-12)
                    AS return_1800s_vol_units,
                log_return_3600s / greatest(volatility_3600s * sqrt(3600.0), 1e-12)
                    AS return_3600s_vol_units,
                volatility_10s / greatest(volatility_300s, 1e-12) AS volatility_10s_vs_300s,
                volatility_60s / greatest(volatility_900s, 1e-12) AS volatility_60s_vs_900s,
                volatility_300s / greatest(volatility_3600s, 1e-12) AS volatility_300s_vs_3600s,
                volatility_900s / greatest(volatility_3600s, 1e-12) AS volatility_900s_vs_3600s,
                spread_bps / greatest(volatility_10s * 10000.0, 1e-12)
                    AS spread_to_10s_volatility,
                normalized_ofi * trade_imbalance AS ofi_trade_flow_agreement,
                ln(1.0 + quote_updates) - ln(1.0 + trade_count)
                    AS quote_trade_log_intensity_gap,
                CASE WHEN high_mid > low_mid
                     THEN 2.0 * (close_mid - low_mid) / (high_mid - low_mid) - 1.0
                     ELSE 0.0 END AS intrasecond_close_location,
                sin(2.0 * pi() * ((second_ms // 1000) % 86400) / 86400.0) AS utc_day_sin,
                cos(2.0 * pi() * ((second_ms // 1000) % 86400) / 86400.0) AS utc_day_cos,
                sin(2.0 * pi() * ((second_ms // 1000) % 28800) / 28800.0) AS funding_cycle_sin,
                cos(2.0 * pi() * ((second_ms // 1000) % 28800) / 28800.0) AS funding_cycle_cos,
                sin(2.0 * pi() * (((second_ms // 1000) + 259200) % 604800) / 604800.0) AS utc_week_sin,
                cos(2.0 * pi() * (((second_ms // 1000) + 259200) % 604800) / 604800.0) AS utc_week_cos,
                CASE WHEN (((second_ms // 1000) // 86400) + 3) % 7 >= 5
                     THEN 1.0 ELSE 0.0 END AS weekend_flag
            FROM temporal
            WHERE observations_60s >= 55
              AND observations_900s >= 840
              AND observations_3600s >= 3360
              AND lag_60_second_ms = second_ms - 60000
              AND previous_second_ms = second_ms - 1000
              AND log_return_1s IS NOT NULL
        )
        SELECT
            d.decision_time_ms,
            d.return_1s_bps, d.return_5s_bps, d.return_15s_bps,
            d.return_30s_bps, d.return_60s_bps, d.return_120s_bps,
            d.return_300s_bps, d.return_900s_bps,
            d.realized_volatility_10s_bps, d.realized_volatility_30s_bps,
            d.realized_volatility_60s_bps, d.realized_volatility_120s_bps,
            d.realized_volatility_300s_bps, d.realized_volatility_900s_bps,
            d.intrasecond_range_bps, d.range_60s_bps, d.range_300s_bps, d.range_900s_bps,
            d.spread_bps, d.max_spread_bps, d.spread_vs_60s_mean, d.spread_vs_300s_mean,
            d.l1_imbalance, d.close_l1_imbalance,
            d.imbalance_10s_mean, d.imbalance_60s_mean, d.imbalance_300s_mean,
             d.microprice_offset_bps, d.normalized_ofi,
             d.ofi_10s_mean, d.ofi_60s_mean, d.ofi_300s_mean,
            d.ofi_delta_5s, d.ofi_delta_15s, d.ofi_delta_30s, d.ofi_delta_60s,
             d.log_quote_updates, d.quote_intensity_vs_60s_mean, d.quote_intensity_vs_300s_mean,
             d.trade_imbalance, d.trade_imbalance_10s_mean, d.trade_imbalance_60s_mean,
            d.trade_imbalance_300s_mean,
            d.trade_imbalance_delta_5s, d.trade_imbalance_delta_15s,
            d.trade_imbalance_delta_30s, d.trade_imbalance_delta_60s,
            d.signed_flow_10s, d.signed_flow_60s, d.signed_flow_300s,
             d.log_base_volume, d.volume_vs_60s_mean, d.volume_vs_300s_mean, d.log_trade_count,
             d.trade_close_vs_mid_bps, d.event_delay_p50_ms, d.event_delay_p99_ms,
             d.event_delay_vs_60s_mean, d.return_efficiency_60s, d.return_efficiency_300s,
            d.l1_imbalance_delta_5s, d.l1_imbalance_delta_15s,
            d.l1_imbalance_delta_30s, d.l1_imbalance_delta_60s,
            d.microprice_delta_5s_bps, d.microprice_delta_15s_bps,
            d.microprice_delta_30s_bps, d.microprice_delta_60s_bps,
            d.return_60s_vol_units, d.return_300s_vol_units, d.return_900s_vol_units,
            d.volatility_10s_vs_300s, d.volatility_60s_vs_900s,
            d.spread_to_10s_volatility, d.ofi_trade_flow_agreement,
            d.quote_trade_log_intensity_gap, d.intrasecond_close_location,
             d.utc_day_sin, d.utc_day_cos, d.funding_cycle_sin, d.funding_cycle_cos,
            d.return_1800s_bps, d.return_3600s_bps,
            d.realized_volatility_1800s_bps, d.realized_volatility_3600s_bps,
            d.range_1800s_bps, d.range_3600s_bps,
            d.spread_vs_900s_mean, d.quote_intensity_vs_900s_mean,
            d.volume_vs_900s_mean, d.trade_intensity_vs_900s_mean,
            d.return_efficiency_900s, d.return_efficiency_3600s,
            d.return_1800s_vol_units, d.return_3600s_vol_units,
            d.volatility_300s_vs_3600s, d.volatility_900s_vs_3600s,
            d.utc_week_sin, d.utc_week_cos, d.weekend_flag,
            (entry_quote.close_ask - entry_quote.close_bid) * 10000.0
                / ((entry_quote.close_ask + entry_quote.close_bid) / 2.0) AS entry_spread_bps,
            (exit_quote.close_ask - exit_quote.close_bid) * 10000.0
                / ((exit_quote.close_ask + exit_quote.close_bid) / 2.0) AS exit_spread_bps,
            d.entry_arrival_ms - entry_quote.last_transaction_time_ms AS entry_quote_age_ms,
            d.exit_arrival_ms - exit_quote.last_transaction_time_ms AS exit_quote_age_ms,
            entry_quote.close_bid AS entry_bid_price,
            entry_quote.close_ask AS entry_ask_price,
            exit_quote.close_bid AS fixed_exit_bid_price,
            exit_quote.close_ask AS fixed_exit_ask_price,
            entry_quote.close_bid_qty AS entry_bid_qty,
            entry_quote.close_ask_qty AS entry_ask_qty,
            exit_quote.close_bid_qty AS fixed_exit_bid_qty,
            exit_quote.close_ask_qty AS fixed_exit_ask_qty
        FROM decisions d
        ASOF LEFT JOIN current_book_ticker_100ms entry_quote
          ON d.symbol = entry_quote.symbol
         AND entry_quote.available_time_ms <= d.entry_arrival_ms
        ASOF LEFT JOIN current_book_ticker_100ms exit_quote
          ON d.symbol = exit_quote.symbol
         AND exit_quote.available_time_ms <= d.exit_arrival_ms
        WHERE entry_quote.last_transaction_time_ms IS NOT NULL
          AND exit_quote.last_transaction_time_ms IS NOT NULL
          AND d.entry_arrival_ms - entry_quote.last_transaction_time_ms BETWEEN 0 AND ?
          AND d.exit_arrival_ms - exit_quote.last_transaction_time_ms BETWEEN 0 AND ?
          AND ((d.decision_time_ms // 1000) % ?) = 0
        ORDER BY d.decision_time_ms
    """
    parameters = [
        MICROSTRUCTURE_TRADE_EMBARGO_MS,
        normalized_symbol,
        lower,
        upper,
        latency,
        latency,
        horizon,
        max_age,
        max_age,
        decision_cadence,
    ]
    cursor = warehouse.connect().execute(sql, parameters)
    values: Mapping[str, np.ndarray] = cursor.fetchnumpy()
    decisions = np.asarray(values["decision_time_ms"], dtype=np.int64)
    if decisions.size == 0:
        feature_values = np.empty((0, len(_FEATURE_COLUMNS)), dtype=np.float32)
    else:
        feature_values = np.column_stack(
            [np.asarray(values[name], dtype=np.float64) for name in _FEATURE_COLUMNS]
        ).astype(np.float32, copy=False)
    entry_bid = np.asarray(values["entry_bid_price"], dtype=np.float64)
    entry_ask = np.asarray(values["entry_ask_price"], dtype=np.float64)
    entry_bid_qty = np.asarray(values["entry_bid_qty"], dtype=np.float64)
    entry_ask_qty = np.asarray(values["entry_ask_qty"], dtype=np.float64)
    exit_bid_qty = np.asarray(values["fixed_exit_bid_qty"], dtype=np.float64)
    exit_ask_qty = np.asarray(values["fixed_exit_ask_qty"], dtype=np.float64)
    fixed_exit_bid = np.asarray(values["fixed_exit_bid_price"], dtype=np.float64)
    fixed_exit_ask = np.asarray(values["fixed_exit_ask_price"], dtype=np.float64)
    long_net_bps, short_net_bps = _net_cross_spread_cash_returns_bps(
        entry_bid,
        entry_ask,
        fixed_exit_bid,
        fixed_exit_ask,
        execution_cost_bps_per_side=fee + additional_slippage,
    )
    long_order_qty = reference_notional / entry_ask
    short_order_qty = reference_notional / entry_bid
    long_participation = np.maximum(
        long_order_qty / entry_ask_qty,
        long_order_qty / exit_bid_qty,
    )
    short_participation = np.maximum(
        short_order_qty / entry_bid_qty,
        short_order_qty / exit_ask_qty,
    )
    dataset = MicrostructureDataset(
        symbol=normalized_symbol,
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=_FEATURE_COLUMNS,
        horizon_seconds=horizon,
        total_latency_ms=latency,
        taker_fee_bps=fee,
        additional_slippage_bps_per_side=additional_slippage,
        reference_order_notional_quote=reference_notional,
        max_l1_participation=participation_limit,
        max_quote_age_ms=max_age,
        decision_cadence_seconds=decision_cadence,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=decisions,
        long_exit_time_ms=decisions + latency + horizon * 1000,
        short_exit_time_ms=decisions + latency + horizon * 1000,
        features=feature_values,
        long_net_bps=long_net_bps,
        short_net_bps=short_net_bps,
        entry_spread_bps=np.asarray(values["entry_spread_bps"], dtype=np.float64),
        exit_spread_bps=np.asarray(values["exit_spread_bps"], dtype=np.float64),
        entry_quote_age_ms=np.asarray(values["entry_quote_age_ms"], dtype=np.int64),
        exit_quote_age_ms=np.asarray(values["exit_quote_age_ms"], dtype=np.int64),
        entry_bid_price=entry_bid,
        entry_ask_price=entry_ask,
        fixed_exit_bid_price=fixed_exit_bid,
        fixed_exit_ask_price=fixed_exit_ask,
        entry_bid_qty=entry_bid_qty,
        entry_ask_qty=entry_ask_qty,
        fixed_exit_bid_qty=exit_bid_qty,
        fixed_exit_ask_qty=exit_ask_qty,
        long_l1_participation=long_participation,
        short_l1_participation=short_participation,
        long_liquidity_eligible=long_participation <= participation_limit,
        short_liquidity_eligible=short_participation <= participation_limit,
        source_evidence=source_evidence,
        trade_feature_embargo_ms=MICROSTRUCTURE_TRADE_EMBARGO_MS,
    )
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: MicrostructureDataset) -> None:
    rows = dataset.rows
    if dataset.features.ndim != 2 or dataset.features.shape[1] != len(dataset.feature_names):
        raise ValueError("microstructure feature matrix shape is inconsistent")
    arrays = (
        dataset.decision_time_ms,
        dataset.long_exit_time_ms,
        dataset.short_exit_time_ms,
        dataset.long_net_bps,
        dataset.short_net_bps,
        dataset.entry_spread_bps,
        dataset.exit_spread_bps,
        dataset.entry_quote_age_ms,
        dataset.exit_quote_age_ms,
        dataset.entry_bid_price,
        dataset.entry_ask_price,
        dataset.fixed_exit_bid_price,
        dataset.fixed_exit_ask_price,
        dataset.entry_bid_qty,
        dataset.entry_ask_qty,
        dataset.fixed_exit_bid_qty,
        dataset.fixed_exit_ask_qty,
        dataset.long_l1_participation,
        dataset.short_l1_participation,
        dataset.long_liquidity_eligible,
        dataset.short_liquidity_eligible,
    )
    if any(len(value) != rows for value in arrays):
        raise ValueError("microstructure dataset arrays have inconsistent lengths")
    if rows and np.any(np.diff(dataset.decision_time_ms) <= 0):
        raise ValueError("microstructure decision timestamps are not strictly increasing")
    earliest_exit = dataset.decision_time_ms + dataset.total_latency_ms
    latest_exit = earliest_exit + dataset.horizon_seconds * 1000 + int(dataset.path_resolution_ms or 0)
    if (
        np.any(dataset.long_exit_time_ms < earliest_exit)
        or np.any(dataset.short_exit_time_ms < earliest_exit)
        or np.any(dataset.long_exit_time_ms > latest_exit)
        or np.any(dataset.short_exit_time_ms > latest_exit)
    ):
        raise ValueError("microstructure exit timestamps fall outside the lifecycle window")
    numeric = (
        dataset.features,
        dataset.long_net_bps,
        dataset.short_net_bps,
        dataset.entry_spread_bps,
        dataset.exit_spread_bps,
        dataset.entry_bid_price,
        dataset.entry_ask_price,
        dataset.fixed_exit_bid_price,
        dataset.fixed_exit_ask_price,
        dataset.entry_bid_qty,
        dataset.entry_ask_qty,
        dataset.fixed_exit_bid_qty,
        dataset.fixed_exit_ask_qty,
        dataset.long_l1_participation,
        dataset.short_l1_participation,
    )
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("microstructure dataset contains non-finite values")
    if np.any(dataset.entry_spread_bps < 0.0) or np.any(dataset.exit_spread_bps < 0.0):
        raise ValueError("microstructure dataset contains crossed execution quotes")
    if (
        not math.isfinite(dataset.reference_order_notional_quote)
        or not math.isfinite(dataset.max_l1_participation)
        or min(
            dataset.reference_order_notional_quote,
            dataset.max_l1_participation,
        )
        <= 0.0
        or dataset.max_l1_participation > 1.0
        or dataset.max_quote_age_ms <= 0
    ):
        raise ValueError("microstructure liquidity contract is invalid")
    if dataset.decision_cadence_seconds <= 0 or dataset.decision_cadence_seconds > 60:
        raise ValueError("microstructure decision cadence is invalid")
    execution_costs = (
        dataset.taker_fee_bps,
        dataset.additional_slippage_bps_per_side,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in execution_costs):
        raise ValueError("microstructure execution-cost contract is invalid")
    quantities = (
        dataset.entry_bid_qty,
        dataset.entry_ask_qty,
        dataset.fixed_exit_bid_qty,
        dataset.fixed_exit_ask_qty,
    )
    if any(np.any(value <= 0.0) for value in quantities):
        raise ValueError("microstructure execution quantities must be positive")
    if np.any(dataset.long_l1_participation <= 0.0) or np.any(
        dataset.short_l1_participation <= 0.0
    ):
        raise ValueError("microstructure L1 participation must be positive")


@dataclass(frozen=True)
class PathTargetEvidence:
    rows: int
    path_resolution_ms: int
    stop_loss_bps: float
    take_profit_bps: float
    trigger_execution_slippage_bps: float
    long_stop_count: int
    long_take_count: int
    long_horizon_count: int
    long_ambiguous_count: int
    short_stop_count: int
    short_take_count: int
    short_horizon_count: int
    short_ambiguous_count: int
    long_mean_holding_seconds: float
    short_mean_holding_seconds: float


def _net_cross_spread_cash_returns_bps(
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    exit_bid: np.ndarray,
    exit_ask: np.ndarray,
    *,
    execution_cost_bps_per_side: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear-contract cash PnL after actual entry/exit notional costs."""

    cost = float(execution_cost_bps_per_side)
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("execution_cost_bps_per_side must be finite and non-negative")
    long_exit_ratio = np.asarray(exit_bid, dtype=np.float64) / np.asarray(
        entry_ask,
        dtype=np.float64,
    )
    short_exit_ratio = np.asarray(exit_ask, dtype=np.float64) / np.asarray(
        entry_bid,
        dtype=np.float64,
    )
    long_net = (
        (long_exit_ratio - 1.0) * 10_000.0
        - cost * (1.0 + long_exit_ratio)
    )
    short_net = (
        (1.0 - short_exit_ratio) * 10_000.0
        - cost * (1.0 + short_exit_ratio)
    )
    return long_net, short_net


@njit(cache=True, parallel=True)
def _first_barrier_outcomes(
    start_indexes: np.ndarray,
    end_indexes: np.ndarray,
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    fixed_exit_bid: np.ndarray,
    fixed_exit_ask: np.ndarray,
    min_bid: np.ndarray,
    max_bid: np.ndarray,
    close_bid: np.ndarray,
    min_ask: np.ndarray,
    max_ask: np.ndarray,
    close_ask: np.ndarray,
    stop_fraction: float,
    take_fraction: float,
    slippage_fraction: float,
    execution_cost_bps_per_side: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(entry_bid)
    long_targets = np.empty(rows, dtype=np.float64)
    short_targets = np.empty(rows, dtype=np.float64)
    long_outcomes = np.zeros(rows, dtype=np.int8)
    short_outcomes = np.zeros(rows, dtype=np.int8)
    long_exit_indexes = end_indexes - 1
    short_exit_indexes = end_indexes - 1
    for index in prange(rows):
        long_exit = fixed_exit_bid[index]
        short_exit = fixed_exit_ask[index]
        long_stop = entry_ask[index] * (1.0 - stop_fraction)
        long_take = entry_ask[index] * (1.0 + take_fraction)
        short_stop = entry_bid[index] * (1.0 + stop_fraction)
        short_take = entry_bid[index] * (1.0 - take_fraction)
        long_open = True
        short_open = True
        for path_index in range(start_indexes[index], end_indexes[index]):
            if long_open:
                hit_stop = min_bid[path_index] <= long_stop
                hit_take = max_bid[path_index] >= long_take
                if hit_stop:
                    long_exit = min_bid[path_index] * (1.0 - slippage_fraction)
                    long_outcomes[index] = 3 if hit_take else 1
                    long_exit_indexes[index] = path_index
                    long_open = False
                elif hit_take:
                    long_exit = close_bid[path_index] * (1.0 - slippage_fraction)
                    long_outcomes[index] = 2
                    long_exit_indexes[index] = path_index
                    long_open = False
            if short_open:
                hit_stop = max_ask[path_index] >= short_stop
                hit_take = min_ask[path_index] <= short_take
                if hit_stop:
                    short_exit = max_ask[path_index] * (1.0 + slippage_fraction)
                    short_outcomes[index] = 3 if hit_take else 1
                    short_exit_indexes[index] = path_index
                    short_open = False
                elif hit_take:
                    short_exit = close_ask[path_index] * (1.0 + slippage_fraction)
                    short_outcomes[index] = 2
                    short_exit_indexes[index] = path_index
                    short_open = False
            if not long_open and not short_open:
                break
        long_exit_ratio = long_exit / entry_ask[index]
        short_exit_ratio = short_exit / entry_bid[index]
        long_targets[index] = (
            (long_exit_ratio - 1.0) * 10000.0
            - execution_cost_bps_per_side * (1.0 + long_exit_ratio)
        )
        short_targets[index] = (
            (1.0 - short_exit_ratio) * 10000.0
            - execution_cost_bps_per_side * (1.0 + short_exit_ratio)
        )
    return (
        long_targets,
        short_targets,
        long_outcomes,
        short_outcomes,
        long_exit_indexes,
        short_exit_indexes,
    )


def _completed_path_index_bounds(
    path_times_ms: np.ndarray,
    entry_arrival_ms: np.ndarray,
    exit_arrival_ms: np.ndarray,
    *,
    resolution_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return path buckets wholly after entry and observable by exit arrival."""

    resolution = int(resolution_ms)
    if resolution <= 0:
        raise ValueError("path resolution must be positive")
    first_complete_start = ((entry_arrival_ms // resolution) + 1) * resolution
    last_complete_start = exit_arrival_ms - resolution
    return (
        np.searchsorted(path_times_ms, first_complete_start, side="left"),
        np.searchsorted(path_times_ms, last_complete_start, side="right"),
    )


def apply_path_aware_lifecycle_targets(
    warehouse: MicrostructureWarehouse,
    dataset: MicrostructureDataset,
    *,
    stop_loss_bps: float,
    take_profit_bps: float,
    trigger_execution_slippage_bps: float,
) -> tuple[MicrostructureDataset, PathTargetEvidence]:
    """Apply adverse-first stop/take triggers with conservative one-second market exits."""

    stop = float(stop_loss_bps)
    take = float(take_profit_bps)
    slippage = float(trigger_execution_slippage_bps)
    if dataset.rows <= 0:
        raise ValueError("path-aware targets require a non-empty dataset")
    if not all(math.isfinite(value) and value > 0.0 for value in (stop, take)):
        raise ValueError("stop_loss_bps and take_profit_bps must be finite and positive")
    if not math.isfinite(slippage) or slippage < 0.0:
        raise ValueError("trigger_execution_slippage_bps must be finite and non-negative")
    first_ms = int(dataset.decision_time_ms[0]) - 2_000
    last_ms = int(dataset.decision_time_ms[-1]) + dataset.horizon_seconds * 1000 + 2_000
    cursor = warehouse.connect().execute(
        """
        SELECT second_ms, min_bid, max_bid, close_bid, min_ask, max_ask, close_ask
        FROM current_book_ticker_path_1s
        WHERE symbol = ? AND second_ms BETWEEN ? AND ?
        ORDER BY second_ms
        """,
        [dataset.symbol, first_ms, last_ms],
    )
    path = cursor.fetchnumpy()
    path_times = np.asarray(path["second_ms"], dtype=np.int64)
    if path_times.size == 0 or np.any(np.diff(path_times) <= 0):
        raise ValueError("path-aware BBO table is missing or not strictly ordered")
    entry_arrival = dataset.decision_time_ms + dataset.total_latency_ms
    exit_arrival = entry_arrival + dataset.horizon_seconds * 1000
    # A path bucket can affect the label only after its full one-second range is
    # observable. Including the bucket containing exit_arrival would look past
    # the requested horizon whenever arrival is not exactly on a second boundary.
    start_indexes, end_indexes = _completed_path_index_bounds(
        path_times,
        entry_arrival,
        exit_arrival,
        resolution_ms=1_000,
    )
    valid = (start_indexes < end_indexes) & (start_indexes < len(path_times)) & (end_indexes > 0)
    if not np.all(valid):
        raise ValueError(f"path-aware BBO coverage missing for {int(np.sum(~valid))} decision rows")
    (
        long_targets,
        short_targets,
        long_outcomes,
        short_outcomes,
        long_exit_indexes,
        short_exit_indexes,
    ) = _first_barrier_outcomes(
        start_indexes.astype(np.int64),
        end_indexes.astype(np.int64),
        dataset.entry_bid_price,
        dataset.entry_ask_price,
        dataset.fixed_exit_bid_price,
        dataset.fixed_exit_ask_price,
        np.asarray(path["min_bid"], dtype=np.float64),
        np.asarray(path["max_bid"], dtype=np.float64),
        np.asarray(path["close_bid"], dtype=np.float64),
        np.asarray(path["min_ask"], dtype=np.float64),
        np.asarray(path["max_ask"], dtype=np.float64),
        np.asarray(path["close_ask"], dtype=np.float64),
        stop / 10000.0,
        take / 10000.0,
        slippage / 10000.0,
        dataset.taker_fee_bps + dataset.additional_slippage_bps_per_side,
    )
    long_exit_times = np.where(
        long_outcomes == 0,
        dataset.long_exit_time_ms,
        path_times[long_exit_indexes] + 1_000,
    ).astype(np.int64)
    short_exit_times = np.where(
        short_outcomes == 0,
        dataset.short_exit_time_ms,
        path_times[short_exit_indexes] + 1_000,
    ).astype(np.int64)
    output = replace(
        dataset,
        target_mode="exchange_trigger_market_exit_1s_adverse_first",
        stop_loss_bps=stop,
        take_profit_bps=take,
        trigger_execution_slippage_bps=slippage,
        path_resolution_ms=1_000,
        long_exit_time_ms=long_exit_times,
        short_exit_time_ms=short_exit_times,
        long_net_bps=long_targets,
        short_net_bps=short_targets,
    )
    _validate_dataset(output)

    def count(values: np.ndarray, code: int) -> int:
        return int(np.sum(values == code))

    evidence = PathTargetEvidence(
        rows=dataset.rows,
        path_resolution_ms=1_000,
        stop_loss_bps=stop,
        take_profit_bps=take,
        trigger_execution_slippage_bps=slippage,
        long_stop_count=count(long_outcomes, 1) + count(long_outcomes, 3),
        long_take_count=count(long_outcomes, 2),
        long_horizon_count=count(long_outcomes, 0),
        long_ambiguous_count=count(long_outcomes, 3),
        short_stop_count=count(short_outcomes, 1) + count(short_outcomes, 3),
        short_take_count=count(short_outcomes, 2),
        short_horizon_count=count(short_outcomes, 0),
        short_ambiguous_count=count(short_outcomes, 3),
        long_mean_holding_seconds=float(
            np.mean(long_exit_times - dataset.decision_time_ms) / 1000.0
        ),
        short_mean_holding_seconds=float(
            np.mean(short_exit_times - dataset.decision_time_ms) / 1000.0
        ),
    )
    return output, evidence


__all__ = [
    "MICROSTRUCTURE_FEATURE_NAMES",
    "MICROSTRUCTURE_TRADE_EMBARGO_MS",
    "MICROSTRUCTURE_FEATURE_VERSION",
    "MicrostructureDataset",
    "PathTargetEvidence",
    "apply_path_aware_lifecycle_targets",
    "build_executable_microstructure_dataset",
]
