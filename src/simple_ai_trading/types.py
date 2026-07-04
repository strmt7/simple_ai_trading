"""Configuration and data structure definitions used across the CLI."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .assets import (
    DEFAULT_CONSERVATIVE_LEVERAGE,
    DEFAULT_MIN_DIVERSIFIED_ASSETS,
    DEFAULT_QUOTE_ASSET,
    DEFAULT_SYMBOL,
    DEFAULT_SYMBOLS,
    MAX_AUTONOMOUS_LEVERAGE,
    normalize_symbol,
    normalize_symbols,
)
from .ai_runtime import AIRuntimeConfig
from .compute import default_compute_backend
from .features import FEATURE_NAMES, FEATURE_VERSION, normalize_enabled_features


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"y", "yes", "true", "1", "on"}:
            return True
        if token in {"n", "no", "false", "0", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _normal_interval(value: object, default: str = "15m") -> str:
    candidate = str(value or "").strip()
    return candidate or default


@dataclass
class RuntimeConfig:
    """Runtime configuration stored in the user profile."""

    symbol: str = DEFAULT_SYMBOL
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    quote_asset: str = DEFAULT_QUOTE_ASSET
    interval: str = "15m"
    market_type: str = "spot"
    testnet: bool = True
    demo: bool = False
    api_key: str = ""
    api_secret: str = ""
    dry_run: bool = True
    validate_account: bool = True
    max_rate_calls_per_minute: int = 1100
    recv_window_ms: int = 5000
    compute_backend: str = field(default_factory=default_compute_backend)
    ai_enabled: bool = True
    ai_provider: str = "auto"
    ai_model: str = "qwen2.5:7b"
    ai_require_gpu: bool = True
    ai_min_free_vram_gb: float = 8.0
    ai_min_free_ram_gb: float = 16.0
    ai_min_model_parameters_b: float = 2.0
    ai_allow_paper_fallback: bool = True
    managed_usdc: float = 0.0
    managed_btc: float = 0.0

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.symbols = normalize_symbols((*normalize_symbols(self.symbols), self.symbol))
        if self.symbol not in self.symbols:
            self.symbols = (self.symbol, *self.symbols)
        self.quote_asset = str(self.quote_asset or DEFAULT_QUOTE_ASSET).strip().upper() or DEFAULT_QUOTE_ASSET
        self.interval = _normal_interval(self.interval)
        self.market_type = str(self.market_type or "spot").lower()
        self.testnet = _coerce_bool(self.testnet, True)
        self.demo = _coerce_bool(self.demo, False)
        self.dry_run = _coerce_bool(self.dry_run, True)
        self.validate_account = _coerce_bool(self.validate_account, True)
        self.api_key = str(self.api_key or "")
        self.api_secret = str(self.api_secret or "")
        self.max_rate_calls_per_minute = max(1, min(2000, _coerce_int(self.max_rate_calls_per_minute, 1100)))
        self.recv_window_ms = max(1, min(60000, _coerce_int(self.recv_window_ms, 5000)))
        self.compute_backend = str(self.compute_backend or default_compute_backend()).strip().lower()
        self.ai_enabled = _coerce_bool(self.ai_enabled, True)
        self.ai_provider = str(self.ai_provider or "auto")
        self.ai_model = str(self.ai_model or "qwen2.5:7b")
        self.ai_require_gpu = _coerce_bool(self.ai_require_gpu, True)
        self.ai_min_free_vram_gb = max(0.0, _finite_float(self.ai_min_free_vram_gb, 8.0))
        self.ai_min_free_ram_gb = max(0.0, _finite_float(self.ai_min_free_ram_gb, 16.0))
        self.ai_min_model_parameters_b = max(0.0, _finite_float(self.ai_min_model_parameters_b, 2.0))
        self.ai_allow_paper_fallback = _coerce_bool(self.ai_allow_paper_fallback, True)
        self.managed_usdc = max(0.0, _finite_float(self.managed_usdc, 0.0))
        self.managed_btc = max(0.0, _finite_float(self.managed_btc, 0.0))

    def asdict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        return payload

    def public_dict(self) -> Dict[str, Any]:
        payload = self.asdict()
        for field_name in ("api_key", "api_secret"):
            if payload.get(field_name):
                payload[field_name] = "<redacted>"
        return payload

    def ai_runtime_config(self) -> AIRuntimeConfig:
        return AIRuntimeConfig(
            enabled=self.ai_enabled,
            provider=self.ai_provider,
            model=self.ai_model,
            require_gpu=self.ai_require_gpu,
            compute_backend=self.compute_backend,
            min_free_vram_gb=self.ai_min_free_vram_gb,
            min_free_ram_gb=self.ai_min_free_ram_gb,
            min_model_parameters_b=self.ai_min_model_parameters_b,
            allow_paper_fallback=self.ai_allow_paper_fallback,
        )


@dataclass
class StrategyConfig:
    """Tunable strategy inputs and risk controls."""

    risk_level: str = "conservative"
    reinvest_profits: bool = False
    leverage: float = DEFAULT_CONSERVATIVE_LEVERAGE
    risk_per_trade: float = 0.003
    max_position_pct: float = 0.08
    max_open_positions: int = DEFAULT_MIN_DIVERSIFIED_ASSETS
    min_diversified_assets: int = DEFAULT_MIN_DIVERSIFIED_ASSETS
    max_asset_allocation_pct: float = 0.20
    max_portfolio_risk_pct: float = 0.015
    min_quote_volume_usdc: float = 50_000_000.0
    min_trade_count_24h: int = 50_000
    max_spread_bps: float = 5.0
    min_liquidity_score: float = 0.80
    unpredictability_cooldown_minutes: int = 90
    max_regime_unpredictability: float = 0.60
    max_prediction_entropy: float = 0.88
    min_model_confidence: float = 0.66
    liquidation_buffer_pct: float = 0.03
    testnet_liquidity_haircut: float = 0.50
    latency_buffer_ms: int = 750
    stop_loss_pct: float = 0.010
    take_profit_pct: float = 0.018
    feature_windows: tuple[int, int] = (10, 40)
    signal_threshold: float = 0.66
    model_lookback: int = 250
    cooldown_minutes: int = 20
    max_trades_per_day: int = 6
    max_drawdown_limit: float = 0.10
    max_daily_loss_pct: float = 0.006
    max_session_loss_pct: float = 0.012
    max_consecutive_losses: int = 2
    max_network_errors: int = 3
    recovery_cooldown_seconds: int = 60
    training_epochs: int = 180
    confidence_beta: float = 0.90
    taker_fee_bps: float = 1.0
    slippage_bps: float = 5.0
    liquidity_risk_enabled: bool = True
    liquidity_lookback_bars: int = 96
    low_liquidity_volume_ratio: float = 0.35
    low_liquidity_signal_threshold_add: float = 0.04
    low_liquidity_size_multiplier: float = 0.50
    dynamic_liquidity_session_enabled: bool = True
    dynamic_liquidity_bucket_minutes: int = 15
    dynamic_liquidity_session_min_samples: int = 8
    low_session_liquidity_volume_ratio: float = 0.45
    low_session_signal_threshold_add: float = 0.01
    low_session_size_multiplier: float = 0.85
    preferred_utc_session_start_hour: int = 13
    preferred_utc_session_end_hour: int = 21
    off_session_signal_threshold_add: float = 0.0
    off_session_size_multiplier: float = 1.0
    label_threshold: float = 0.001
    feature_version: str = FEATURE_VERSION
    enabled_features: tuple[str, ...] = FEATURE_NAMES
    order_type: str = "MARKET"
    time_in_force: str = "GTC"
    post_only: bool = False
    reduce_only_on_close: bool = True
    external_signals_enabled: bool = False
    external_signal_max_adjustment: float = 0.04
    external_signal_min_providers: int = 2
    external_signal_ttl_seconds: int = 60
    external_signal_timeout_seconds: float = 3.0
    external_signal_news_provider_limit: int = 93
    external_signal_news_items_per_provider: int = 4
    external_signal_provider_parallelism: int = 24
    external_signal_provider_jitter_seconds: float = 0.25
    external_signal_poll_jitter_seconds: float = 2.0
    external_signal_short_reaction_refresh_seconds: int = 30
    external_news_ai_enabled: bool = False
    external_news_ai_model: str = "gemma4:e4b"
    external_news_ai_url: str = "http://127.0.0.1:11434"
    external_news_ai_timeout_seconds: float = 3.0
    telemetry_enabled: bool = True
    telemetry_db_path: str = "data/trading_telemetry.sqlite"
    source_grading_enabled: bool = True
    source_grading_interval_seconds: int = 3600
    source_grading_window_hours: int = 24
    source_grade_max_age_hours: float = 168.0

    def __post_init__(self) -> None:
        windows = tuple(self.feature_windows) if isinstance(self.feature_windows, (list, tuple)) else (10, 40)
        if len(windows) != 2:
            windows = (10, 40)
        short_window = max(1, _coerce_int(windows[0], 10))
        long_window = max(short_window, _coerce_int(windows[1], 40))
        self.feature_windows = (short_window, long_window)
        self.risk_level = str(self.risk_level or "conservative").strip().lower()
        if self.risk_level not in {"conservative", "regular", "aggressive"}:
            self.risk_level = "conservative"
        self.reinvest_profits = _coerce_bool(self.reinvest_profits, False)
        self.leverage = max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, _finite_float(self.leverage, 1.0)))
        self.risk_per_trade = min(1.0, max(0.0, _finite_float(self.risk_per_trade, 0.003)))
        self.max_position_pct = min(1.0, max(0.0, _finite_float(self.max_position_pct, 0.08)))
        self.max_open_positions = max(0, _coerce_int(self.max_open_positions, DEFAULT_MIN_DIVERSIFIED_ASSETS))
        self.min_diversified_assets = max(1, _coerce_int(self.min_diversified_assets, DEFAULT_MIN_DIVERSIFIED_ASSETS))
        self.max_asset_allocation_pct = min(1.0, max(0.01, _finite_float(self.max_asset_allocation_pct, 0.20)))
        self.max_portfolio_risk_pct = min(1.0, max(0.0, _finite_float(self.max_portfolio_risk_pct, 0.015)))
        self.min_quote_volume_usdc = max(0.0, _finite_float(self.min_quote_volume_usdc, 50_000_000.0))
        self.min_trade_count_24h = max(0, _coerce_int(self.min_trade_count_24h, 50_000))
        self.max_spread_bps = max(0.0, _finite_float(self.max_spread_bps, 5.0))
        self.min_liquidity_score = min(1.0, max(0.0, _finite_float(self.min_liquidity_score, 0.80)))
        self.unpredictability_cooldown_minutes = max(0, _coerce_int(self.unpredictability_cooldown_minutes, 90))
        self.max_regime_unpredictability = min(1.0, max(0.0, _finite_float(self.max_regime_unpredictability, 0.60)))
        self.max_prediction_entropy = min(1.0, max(0.0, _finite_float(self.max_prediction_entropy, 0.97)))
        self.min_model_confidence = min(1.0, max(0.0, _finite_float(self.min_model_confidence, 0.66)))
        self.liquidation_buffer_pct = min(1.0, max(0.0, _finite_float(self.liquidation_buffer_pct, 0.03)))
        self.testnet_liquidity_haircut = min(1.0, max(0.0, _finite_float(self.testnet_liquidity_haircut, 0.50)))
        self.latency_buffer_ms = max(0, _coerce_int(self.latency_buffer_ms, 750))
        self.stop_loss_pct = _finite_float(self.stop_loss_pct, 0.010)
        self.take_profit_pct = _finite_float(self.take_profit_pct, 0.018)
        self.signal_threshold = min(0.99, max(0.01, _finite_float(self.signal_threshold, 0.66)))
        self.model_lookback = max(1, _coerce_int(self.model_lookback, 250))
        self.cooldown_minutes = max(0, _coerce_int(self.cooldown_minutes, 20))
        self.max_trades_per_day = max(0, _coerce_int(self.max_trades_per_day, 6))
        self.max_drawdown_limit = _finite_float(self.max_drawdown_limit, 0.10)
        self.max_daily_loss_pct = min(0.25, max(0.0, _finite_float(self.max_daily_loss_pct, 0.006)))
        self.max_session_loss_pct = min(0.50, max(0.0, _finite_float(self.max_session_loss_pct, 0.012)))
        self.max_consecutive_losses = max(0, min(100, _coerce_int(self.max_consecutive_losses, 2)))
        self.max_network_errors = max(1, min(100, _coerce_int(self.max_network_errors, 3)))
        self.recovery_cooldown_seconds = max(0, min(3600, _coerce_int(self.recovery_cooldown_seconds, 60)))
        self.training_epochs = max(1, _coerce_int(self.training_epochs, 180))
        self.confidence_beta = min(1.0, max(0.0, _finite_float(self.confidence_beta, 0.90)))
        self.taker_fee_bps = _finite_float(self.taker_fee_bps, 1.0)
        self.slippage_bps = _finite_float(self.slippage_bps, 5.0)
        self.liquidity_risk_enabled = _coerce_bool(self.liquidity_risk_enabled, True)
        self.liquidity_lookback_bars = max(8, min(10_000, _coerce_int(self.liquidity_lookback_bars, 96)))
        self.low_liquidity_volume_ratio = min(1.0, max(0.01, _finite_float(self.low_liquidity_volume_ratio, 0.35)))
        self.low_liquidity_signal_threshold_add = min(0.25, max(0.0, _finite_float(self.low_liquidity_signal_threshold_add, 0.04)))
        self.low_liquidity_size_multiplier = min(1.0, max(0.05, _finite_float(self.low_liquidity_size_multiplier, 0.50)))
        self.dynamic_liquidity_session_enabled = _coerce_bool(self.dynamic_liquidity_session_enabled, True)
        self.dynamic_liquidity_bucket_minutes = max(1, min(60, _coerce_int(self.dynamic_liquidity_bucket_minutes, 15)))
        self.dynamic_liquidity_session_min_samples = max(3, min(10_000, _coerce_int(self.dynamic_liquidity_session_min_samples, 8)))
        self.low_session_liquidity_volume_ratio = min(1.0, max(0.01, _finite_float(self.low_session_liquidity_volume_ratio, 0.45)))
        self.low_session_signal_threshold_add = min(0.10, max(0.0, _finite_float(self.low_session_signal_threshold_add, 0.01)))
        self.low_session_size_multiplier = min(1.0, max(0.10, _finite_float(self.low_session_size_multiplier, 0.85)))
        self.preferred_utc_session_start_hour = max(0, min(23, _coerce_int(self.preferred_utc_session_start_hour, 13)))
        self.preferred_utc_session_end_hour = max(0, min(24, _coerce_int(self.preferred_utc_session_end_hour, 21)))
        self.off_session_signal_threshold_add = min(0.10, max(0.0, _finite_float(self.off_session_signal_threshold_add, 0.0)))
        self.off_session_size_multiplier = min(1.0, max(0.10, _finite_float(self.off_session_size_multiplier, 1.0)))
        self.label_threshold = _finite_float(self.label_threshold, 0.001)
        self.feature_version = str(self.feature_version or FEATURE_VERSION)
        self.order_type = str(self.order_type or "MARKET").upper()
        self.time_in_force = str(self.time_in_force or "GTC").upper()
        self.post_only = _coerce_bool(self.post_only, False)
        self.reduce_only_on_close = _coerce_bool(self.reduce_only_on_close, True)
        self.external_signals_enabled = _coerce_bool(self.external_signals_enabled, False)
        self.external_signal_max_adjustment = min(
            0.20,
            max(0.0, _finite_float(self.external_signal_max_adjustment, 0.04)),
        )
        self.external_signal_min_providers = max(0, _coerce_int(self.external_signal_min_providers, 2))
        self.external_signal_ttl_seconds = max(0, _coerce_int(self.external_signal_ttl_seconds, 60))
        self.external_signal_timeout_seconds = min(
            30.0,
            max(0.1, _finite_float(self.external_signal_timeout_seconds, 3.0)),
        )
        self.external_signal_news_provider_limit = max(0, _coerce_int(self.external_signal_news_provider_limit, 93))
        self.external_signal_news_items_per_provider = max(1, min(10, _coerce_int(self.external_signal_news_items_per_provider, 4)))
        self.external_signal_provider_parallelism = max(1, min(64, _coerce_int(self.external_signal_provider_parallelism, 24)))
        self.external_signal_provider_jitter_seconds = min(
            30.0,
            max(0.0, _finite_float(self.external_signal_provider_jitter_seconds, 0.25)),
        )
        self.external_signal_poll_jitter_seconds = min(
            60.0,
            max(0.0, _finite_float(self.external_signal_poll_jitter_seconds, 2.0)),
        )
        self.external_signal_short_reaction_refresh_seconds = max(
            1,
            _coerce_int(self.external_signal_short_reaction_refresh_seconds, 30),
        )
        self.external_news_ai_enabled = _coerce_bool(self.external_news_ai_enabled, False)
        self.external_news_ai_model = str(self.external_news_ai_model or "gemma4:e4b")
        self.external_news_ai_url = str(self.external_news_ai_url or "http://127.0.0.1:11434")
        self.external_news_ai_timeout_seconds = min(
            30.0,
            max(0.1, _finite_float(self.external_news_ai_timeout_seconds, 3.0)),
        )
        self.telemetry_enabled = _coerce_bool(self.telemetry_enabled, True)
        self.telemetry_db_path = str(self.telemetry_db_path or "data/trading_telemetry.sqlite")
        self.source_grading_enabled = _coerce_bool(self.source_grading_enabled, True)
        self.source_grading_interval_seconds = max(60, _coerce_int(self.source_grading_interval_seconds, 3600))
        self.source_grading_window_hours = max(1, _coerce_int(self.source_grading_window_hours, 24))
        self.source_grade_max_age_hours = min(
            8760.0,
            max(0.0, _finite_float(self.source_grade_max_age_hours, 168.0)),
        )
        self.enabled_features = normalize_enabled_features(self.enabled_features)

    def asdict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["feature_windows"] = list(self.feature_windows)
        payload["enabled_features"] = list(self.enabled_features)
        return payload


@dataclass
class RiskProfile:
    """Runtime stats and constraints that are not part of strategy tuning."""

    starting_cash: float = 1000.0
    max_daily_trades: int = 50
    max_open_positions: int = 1
    last_run: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


def config_paths() -> Dict[str, Path]:
    """Return the default config directories used by the CLI."""

    base_home = Path(os.environ.get("HOME") or Path.home())
    base = base_home / ".config" / "simple_ai_trading"
    return {
        "base": base,
        "runtime": base / "runtime.json",
        "strategy": base / "strategy.json",
    }
