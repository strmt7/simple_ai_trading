"""Entry point for the Simple AI Trading BTC/ETH/SOL day-trading CLI."""

from __future__ import annotations

import argparse
import asyncio
import builtins
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import random
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar, cast

from .api import BinanceAPIError, BinanceClient, Candle
from .api_budget import (
    api_budget_startup_block_reason,
    build_api_budget_report,
    render_api_budget,
    summarize_api_budget,
)
from .ai_runtime import detect_ai_capabilities, render_ai_capability_report
from .advanced_model import (
    AdvancedFeatureConfig,
    advanced_config_from_signature,
    advanced_feature_signature,
    default_config_for,
    make_advanced_inference_rows,
    make_advanced_rows,
)
from .assets import (
    DEFAULT_AGGRESSIVE_LEVERAGE,
    DEFAULT_CONSERVATIVE_LEVERAGE,
    DEFAULT_REGULAR_LEVERAGE,
    MAX_AUTONOMOUS_LEVERAGE,
    SUPPORTED_MAJOR_QUOTE_ASSETS,
    is_supported_major_symbol,
    normalize_symbol,
    symbol_base_for_supported_quote,
)
from .backtest import calibrate_threshold_for_backtest, risk_adjusted_backtest_score, run_backtest
from .binance_archive import (
    archive_listing_items_by_url,
    archive_url_period,
    filter_archive_urls_by_period,
    ingest_archive_urls,
    list_archive_items,
    list_archive_urls,
    validate_archive_period_window,
)
from .compute import BackendInfo, default_compute_backend, describe_backend, resolve_backend
from .commission import apply_offline_commission_floor, apply_verified_commission_rate
from .config import config_paths, load_runtime, load_strategy, prompt_runtime, save_runtime, save_strategy
from .dashboard import DashboardSnapshot, load_artifact_preview, render_dashboard
from . import data_workflows
from .data_coverage import describe_candle_coverage
from .data_downloader import MarketDataSyncConfig, render_sync_result, sync_market_data
from .features import FEATURE_NAMES, ModelRow, feature_signature, make_inference_rows, make_rows, normalize_enabled_features
from .api import SymbolConstraints
from .external_signals import ExternalSignalReport, collect_external_signals, render_external_signal_report
from .intervals import interval_milliseconds
from .liquidity_session import apply_liquidity_session_meta, liquidity_session_adjustment
from .live_artifacts import build_live_run_payload
from .market_data import clean_candles
from .market_store import MarketDataStore
from .microstructure_data import capture_binance_futures_microstructure
from .microstructure_runtime import MICROSTRUCTURE_STREAM_WARMUP_SECONDS
from .optimization_evidence import select_top_liquidity_symbols
from .meta_label import apply_meta_label_policy
from .position_lifecycle import evaluate_position_exit
from .model import (
    assess_probability_calibration,
    build_model_quality_report,
    calibrate_threshold,
    calibrate_probability_temperature,
    confidence_adjusted_probability,
    evaluate_classification,
    evaluate,
    feature_drift_report,
    ModelFeatureMismatchError,
    ModelLoadError,
    load_model,
    market_direction_from_probability,
    model_decision_threshold,
    model_direction_thresholds,
    serialize_model,
    temporal_validation_split,
    train,
    TrainedModel,
    walk_forward_report,
)
from .model_readiness import ModelPromotionError, assert_model_promoted
from .objective import available_objectives
from .positions import (
    ClosedTrade,
    OpenPosition,
    PositionsStore,
    bot_client_order_id,
    bot_ownership_rejection_reason,
    new_position_id,
)
from .polymarket_paper import PolymarketPaperBroker, PolymarketPaperCoordinator
from .polymarket_paper_plan import (
    build_polymarket_paper_plan,
    run_polymarket_paper_plan,
)
from .polymarket_ai_veto import (
    PolymarketAIVetoConfig,
    benchmark_polymarket_ai_veto,
    build_polymarket_ai_veto_cases,
)
from .polymarket_action_pipeline import (
    PolymarketActionPipelineConfig,
    materialize_polymarket_action_value_batches,
)
from .polymarket_ridge import (
    fit_and_evaluate_polymarket_ridge,
    load_polymarket_ridge_dataset,
    materialize_polymarket_ridge_report,
)
from .polymarket_coverage import inspect_polymarket_feed_coverage
from .polymarket_continuity import evaluate_polymarket_continuity_eligibility
from .polymarket_features import (
    PolymarketFeatureConfig,
    build_polymarket_feature_dataset,
    materialize_polymarket_feature_dataset,
)
from .polymarket_model import (
    POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION,
    POLYMARKET_PROFILE_CONTRACT_SHA256,
    PolymarketModelConfig,
    build_polymarket_model_dataset,
    fit_polymarket_offset_model,
    fit_polymarket_profile_challenger,
    predict_polymarket_probabilities,
    predict_polymarket_profile_probabilities,
    split_polymarket_model_dataset,
)
from .polymarket_model_execution import (
    POLYMARKET_RETRY_CHALLENGER_SCHEMA_VERSION,
    POLYMARKET_RETRY_CONTRACT_SHA256,
    PolymarketExecutionResearchConfig,
    build_polymarket_policy_selection,
    evaluate_polymarket_execution_policy,
    evaluate_polymarket_retry_execution_policy,
)
from .polymarket_publication import (
    POLYMARKET_MODEL_ARTIFACT_SCHEMA_VERSION,
    publish_polymarket_model_artifact,
)
from .polymarket_recorder import PolymarketEvidenceStore, PolymarketPublicRecorder
from .polymarket_replay import PolymarketEvidenceReplay
from .polymarket_resolution import PolymarketResolutionFinalizer
from .polymarket_source_verification import (
    verify_polymarket_model_artifact_source,
)
from .reconciliation import reconcile_account_positions
from .risk_controls import (
    EntryRiskDecision,
    assess_entry_risk,
    build_risk_policy_report,
    market_regime_unpredictability,
    regime_unpredictability_requires_cooldown,
    render_risk_policy_report,
    stop_loss_sized_notional_pct,
)
from .regime import classify_market_regime
from . import risk_workflows
from .strategy_overrides import apply_model_strategy_overrides
from .storage import write_json_atomic
from .source_grading import grade_sources, render_source_grade_run
from .style import supports_ansi_terminal
from .types import RuntimeConfig, StrategyConfig

_JITTER_RANDOM = random.SystemRandom()
_SEGMENTED_GAPS_HELP = (
    "admit only continuity segments that reset CLOB, direct Binance, and RTDS "
    "state after a hash-audited reconnect"
)


_TRAINING_PRESETS: dict[str, dict[str, object]] = {
    "custom": {},
    "quick": {
        "epochs": 80,
        "walk_forward": False,
        "calibrate_threshold": False,
    },
    "balanced": {
        "epochs": 180,
        "walk_forward": True,
        "walk_forward_train": 300,
        "walk_forward_test": 60,
        "walk_forward_step": 30,
        "calibrate_threshold": True,
    },
    "thorough": {
        "epochs": 350,
        "walk_forward": True,
        "walk_forward_train": 360,
        "walk_forward_test": 90,
        "walk_forward_step": 30,
        "calibrate_threshold": True,
    },
}
_T = TypeVar("_T")


_STRATEGY_PROFILES: dict[str, dict[str, object]] = {
    "custom": {},
    "conservative": {
        "risk_level": "conservative",
        "leverage": DEFAULT_CONSERVATIVE_LEVERAGE,
        "risk_per_trade": 0.003,
        "max_position_pct": 0.08,
        "max_asset_allocation_pct": 0.20,
        "max_portfolio_risk_pct": 0.015,
        "stop_loss_pct": 0.010,
        "take_profit_pct": 0.018,
        "cooldown_minutes": 20,
        "unpredictability_cooldown_minutes": 90,
        "max_open_positions": 3,
        "min_diversified_assets": 3,
        "max_trades_per_day": 6,
        "signal_threshold": 0.66,
        "min_model_confidence": 0.66,
        "max_prediction_entropy": 0.88,
        "max_drawdown_limit": 0.10,
        "max_daily_loss_pct": 0.006,
        "max_session_loss_pct": 0.012,
        "max_consecutive_losses": 2,
        "max_network_errors": 3,
        "recovery_cooldown_seconds": 60,
        "min_quote_volume_usdc": 50_000_000.0,
        "min_trade_count_24h": 50_000,
        "max_spread_bps": 5.0,
        "min_liquidity_score": 0.80,
        "max_regime_unpredictability": 0.60,
        "training_epochs": 180,
        "confidence_beta": 0.90,
        "external_signals_enabled": True,
        "external_signal_max_adjustment": 0.03,
        "external_signal_min_providers": 2,
    },
    "regular": {
        "risk_level": "regular",
        "leverage": DEFAULT_REGULAR_LEVERAGE,
        "risk_per_trade": 0.006,
        "max_position_pct": 0.15,
        "max_asset_allocation_pct": 0.25,
        "max_portfolio_risk_pct": 0.03,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.03,
        "cooldown_minutes": 10,
        "unpredictability_cooldown_minutes": 45,
        "max_open_positions": 4,
        "min_diversified_assets": 3,
        "max_trades_per_day": 12,
        "signal_threshold": 0.58,
        "min_model_confidence": 0.58,
        "max_prediction_entropy": 0.94,
        "max_drawdown_limit": 0.18,
        "max_daily_loss_pct": 0.010,
        "max_session_loss_pct": 0.020,
        "max_consecutive_losses": 3,
        "max_network_errors": 3,
        "recovery_cooldown_seconds": 45,
        "min_quote_volume_usdc": 25_000_000.0,
        "min_trade_count_24h": 25_000,
        "max_spread_bps": 8.0,
        "min_liquidity_score": 0.70,
        "max_regime_unpredictability": 0.72,
        "training_epochs": 250,
        "confidence_beta": 0.85,
        "external_signals_enabled": True,
        "external_signal_max_adjustment": 0.04,
        "external_signal_min_providers": 2,
    },
    "aggressive": {
        "risk_level": "aggressive",
        "leverage": DEFAULT_AGGRESSIVE_LEVERAGE,
        "risk_per_trade": 0.010,
        "max_position_pct": 0.20,
        "max_asset_allocation_pct": 0.30,
        "max_portfolio_risk_pct": 0.05,
        "stop_loss_pct": 0.025,
        "take_profit_pct": 0.04,
        "cooldown_minutes": 5,
        "unpredictability_cooldown_minutes": 20,
        "max_open_positions": 5,
        "min_diversified_assets": 3,
        "max_trades_per_day": 24,
        "signal_threshold": 0.55,
        "min_model_confidence": 0.55,
        "max_prediction_entropy": 0.97,
        "max_drawdown_limit": 0.25,
        "max_daily_loss_pct": 0.015,
        "max_session_loss_pct": 0.030,
        "max_consecutive_losses": 4,
        "max_network_errors": 4,
        "recovery_cooldown_seconds": 30,
        "min_quote_volume_usdc": 15_000_000.0,
        "min_trade_count_24h": 15_000,
        "max_spread_bps": 12.0,
        "min_liquidity_score": 0.60,
        "max_regime_unpredictability": 0.85,
        "training_epochs": 300,
        "confidence_beta": 0.80,
        "external_signals_enabled": True,
        "external_signal_max_adjustment": 0.05,
        "external_signal_min_providers": 2,
    },
}

_STRATEGY_PROFILES["balanced"] = dict(_STRATEGY_PROFILES["regular"])
_STRATEGY_PROFILES["active"] = dict(_STRATEGY_PROFILES["aggressive"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-ai-trading",
        description="Autonomous BTC/ETH/SOL non-mainnet trading CLI for Binance (spot + futures).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_config = subparsers.add_parser("configure", help="configure Binance credentials and defaults")
    parser_config.set_defaults(func=command_configure)

    parser_connect = subparsers.add_parser("connect", help="validate credentials and connectivity")
    parser_connect.set_defaults(func=command_connect)

    parser_roundtrip = subparsers.add_parser(
        "spot-roundtrip",
        help="place a tiny signed spot testnet/demo roundtrip order with balance and filter prechecks",
    )
    parser_roundtrip.add_argument("--quantity", type=float, default=0.00008, help="base-asset quantity to test")
    parser_roundtrip.add_argument(
        "--mode",
        choices=["auto", "buy-sell", "sell-buy"],
        default="auto",
        help="order sequence; auto buys first when quote balance is available, otherwise sells first when base balance is available",
    )
    parser_roundtrip.add_argument("--yes", action="store_true", help="confirm signed testnet/demo order placement")
    parser_roundtrip.set_defaults(func=command_spot_roundtrip)

    parser_doctor = subparsers.add_parser(
        "doctor",
        help="run local readiness checks before paper or non-mainnet trading",
    )
    parser_doctor.add_argument("--input", default="data/historical_market.json")
    parser_doctor.add_argument("--model", default="data/model.json")
    parser_doctor.add_argument("--online", action="store_true", help="also check exchange connectivity")
    parser_doctor.set_defaults(func=command_doctor)

    parser_audit = subparsers.add_parser("audit", help="run local data/model/risk diagnostics without network calls")
    parser_audit.add_argument("--input", default="data/historical_market.json")
    parser_audit.add_argument("--model", default="data/model.json")
    parser_audit.set_defaults(func=command_audit)

    parser_risk = subparsers.add_parser("risk", help="show local risk policy before paper or live trading")
    parser_risk.add_argument("--model", default="data/model.json")
    parser_risk.add_argument("--paper", action="store_true", help="assess paper/dry-run execution")
    parser_risk.add_argument("--live", action="store_true", help="assess authenticated testnet/demo execution")
    parser_risk.add_argument("--leverage", type=float, default=None, help="optional futures leverage override")
    parser_risk.add_argument("--json", action="store_true")
    parser_risk.set_defaults(func=command_risk)

    parser_reconcile = subparsers.add_parser(
        "reconcile",
        help="compare signed exchange exposure with the local autonomous position ledger",
    )
    parser_reconcile.add_argument("--json", action="store_true")
    parser_reconcile.add_argument("--output", default="data/autonomous/reconciliation.json")
    parser_reconcile.add_argument("--quantity-tolerance", type=float, default=1e-8)
    parser_reconcile.set_defaults(func=command_reconcile)

    parser_universe = subparsers.add_parser("universe", help="measure BTC/ETH/SOL high-liquidity eligibility")
    parser_universe.add_argument("--symbols", default=None, help="comma-separated symbols; default uses runtime.symbols")
    parser_universe.add_argument("--json", action="store_true")
    parser_universe.set_defaults(func=command_universe)

    parser_report = subparsers.add_parser("report", help="show dashboard, artifacts, and optional readiness checks")
    parser_report.add_argument("--account", action="store_true", help="include authenticated account state")
    parser_report.add_argument("--doctor", action="store_true", help="include readiness checks")
    parser_report.add_argument("--no-doctor", action="store_false", dest="doctor", help="omit readiness checks")
    parser_report.add_argument("--online", action="store_true", help="include exchange connectivity in readiness checks")
    parser_report.add_argument("--input", default="data/historical_market.json")
    parser_report.add_argument("--model", default="data/model.json")
    parser_report.set_defaults(doctor=True)
    parser_report.set_defaults(func=command_report)

    parser_coordinator = subparsers.add_parser(
        "coordinator",
        help="show independent risk, ML, AI, learning, reconciliation, and execution loop state",
    )
    parser_coordinator.add_argument("--model", default="data/model.json")
    parser_coordinator.add_argument("--positions-root", default="data/autonomous")
    parser_coordinator.add_argument("--json", action="store_true")
    parser_coordinator.set_defaults(func=command_coordinator)

    parser_menu = subparsers.add_parser("menu", help="launch the full-screen operator console")
    parser_menu.set_defaults(func=command_menu)

    parser_fetch = subparsers.add_parser("fetch", help="download symbol klines")
    parser_fetch.add_argument("--symbol", default=None)
    parser_fetch.add_argument("--interval", default=None)
    parser_fetch.add_argument("--limit", type=int, default=500)
    parser_fetch.add_argument("--batch-size", type=int, default=1000, help="klines per request (spot max 1000, futures max 1500)")
    parser_fetch.add_argument("--output", default="data/historical_market.json")
    parser_fetch.set_defaults(func=command_fetch)

    parser_data_sync = subparsers.add_parser(
        "data-sync",
        help="rate-limited Binance downloader for candles and auxiliary metrics into SQLite",
    )
    parser_data_sync.add_argument("--db", default="data/market_data.sqlite")
    parser_data_sync.add_argument("--symbol", default=None)
    parser_data_sync.add_argument("--interval", default=None)
    parser_data_sync.add_argument("--market", choices=["spot", "futures"], default=None)
    parser_data_sync.add_argument("--rows", type=int, default=500)
    parser_data_sync.add_argument(
        "--full-history",
        action="store_true",
        help="page historical klines backward until the exchange has no older closed candles",
    )
    parser_data_sync.add_argument("--batch-size", type=int, default=1000)
    parser_data_sync.add_argument("--include-futures-metrics", action="store_true", default=True)
    parser_data_sync.add_argument("--no-include-futures-metrics", action="store_false", dest="include_futures_metrics")
    parser_data_sync.add_argument("--loop", action="store_true", help="keep syncing in the foreground")
    parser_data_sync.add_argument("--iterations", type=int, default=1, help="foreground loop iterations; 0 means unlimited")
    parser_data_sync.add_argument("--sleep", type=int, default=300, help="seconds between loop iterations")
    parser_data_sync.add_argument("--background", action="store_true", help="start a detached downloader process")
    parser_data_sync.add_argument("--pid-file", default="data/market_data_sync.pid")
    parser_data_sync.add_argument("--log-file", default="data/market_data_sync.log")
    parser_data_sync.add_argument("--json", action="store_true")
    parser_data_sync.set_defaults(func=command_data_sync)

    parser_data_health = subparsers.add_parser(
        "data-health",
        help="audit SQLite market-data coverage, gaps, archive files, and checksum evidence",
    )
    parser_data_health.add_argument("--db", default="data/market_data.sqlite")
    parser_data_health.add_argument("--symbol", default=None)
    parser_data_health.add_argument("--symbols", default=None, help="comma-separated symbols; defaults to stored series")
    parser_data_health.add_argument("--interval", default=None)
    parser_data_health.add_argument("--market", choices=["spot", "futures"], default=None)
    parser_data_health.add_argument("--min-rows", type=int, default=0)
    parser_data_health.add_argument("--min-coverage-ratio", type=float, default=0.995)
    parser_data_health.add_argument("--max-gap-count", type=int, default=0)
    parser_data_health.add_argument("--require-verified-checksum", action="store_true")
    parser_data_health.add_argument("--json", action="store_true")
    parser_data_health.set_defaults(func=command_data_health)

    parser_api_budget = subparsers.add_parser(
        "api-budget",
        help="show cached or refreshed Binance API used-weight and order-count budget",
    )
    parser_api_budget.add_argument("--db", default="data/market_data.sqlite")
    parser_api_budget.add_argument("--market", choices=["spot", "futures"], default=None)
    parser_api_budget.add_argument("--refresh", action="store_true", help="query Binance exchangeInfo once and cache the latest headers")
    parser_api_budget.add_argument("--cached-only", action="store_true", help="do not refresh even when the cached sample is stale")
    parser_api_budget.add_argument("--max-age-seconds", type=int, default=90, help="automatic refresh threshold for cached status")
    parser_api_budget.add_argument("--compact", action="store_true", help="print one status-bar friendly line")
    parser_api_budget.add_argument("--json", action="store_true")
    parser_api_budget.set_defaults(func=command_api_budget)

    parser_polymarket_record = subparsers.add_parser(
        "polymarket-record",
        help="record prospective BTC/ETH/SOL 5-minute public evidence for paper trading",
        description=(
            "Record public Polymarket CLOB/RTDS and direct Binance streams into a "
            "single audit-ready DuckDB database. This command never authenticates or "
            "places an order."
        ),
    )
    parser_polymarket_record.add_argument("--database", default="data/polymarket-paper.duckdb")
    parser_polymarket_record.add_argument("--duration-seconds", type=int, default=300)
    parser_polymarket_record.add_argument("--discovery-interval-seconds", type=int, default=60)
    parser_polymarket_record.add_argument("--queue-capacity", type=int, default=100_000)
    parser_polymarket_record.add_argument("--memory-limit", default="4GB")
    parser_polymarket_record.add_argument("--database-threads", type=int, default=2)
    parser_polymarket_record.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=30,
        help="bounded capture and integrity-audit progress cadence",
    )
    parser_polymarket_record.add_argument(
        "--progress-path",
        default=None,
        help="optional atomic JSON sidecar for CLI/app progress",
    )
    parser_polymarket_record.add_argument("--json", action="store_true")
    parser_polymarket_record.set_defaults(func=command_polymarket_record)

    parser_polymarket_resolve = subparsers.add_parser(
        "polymarket-resolve",
        help="finalize BTC/ETH/SOL 5-minute paper labels from official public sources",
        description=(
            "Persist an outcome only after the official CLOB and Gamma APIs are both "
            "terminal and agree exactly. This command never authenticates or places "
            "an order."
        ),
    )
    parser_polymarket_resolve.add_argument(
        "--database", default="data/polymarket-paper.duckdb"
    )
    parser_polymarket_resolve.add_argument("--run-id", default=None)
    parser_polymarket_resolve.add_argument("--wait-seconds", type=int, default=0)
    parser_polymarket_resolve.add_argument(
        "--poll-interval-seconds", type=int, default=15
    )
    parser_polymarket_resolve.add_argument("--memory-limit", default="1GB")
    parser_polymarket_resolve.add_argument("--database-threads", type=int, default=2)
    parser_polymarket_resolve.add_argument("--json", action="store_true")
    parser_polymarket_resolve.set_defaults(func=command_polymarket_resolve)

    parser_polymarket_features = subparsers.add_parser(
        "polymarket-features",
        help="build leakage-safe BTC/ETH/SOL 5-minute model features",
        description=(
            "Build and materialize hash-bound decision-time features from one "
            "validated prospective Polymarket recorder run. Strict gap-free replay "
            "is the default. Official outcomes are attached only as future labels; "
            "unresolved rows remain shadow-only."
        ),
    )
    parser_polymarket_features.add_argument(
        "--database", default="data/polymarket-paper.duckdb"
    )
    parser_polymarket_features.add_argument("--run-id", default=None)
    parser_polymarket_features.add_argument("--cadence-ms", type=int, default=250)
    parser_polymarket_features.add_argument("--warmup-ms", type=int, default=5_000)
    parser_polymarket_features.add_argument(
        "--minimum-resolved-markets-per-asset", type=int, default=30
    )
    parser_polymarket_features.add_argument(
        "--allow-segmented-gaps",
        action="store_true",
        help=_SEGMENTED_GAPS_HELP,
    )
    parser_polymarket_features.add_argument("--memory-limit", default="1GB")
    parser_polymarket_features.add_argument("--database-threads", type=int, default=2)
    parser_polymarket_features.add_argument("--json", action="store_true")
    parser_polymarket_features.set_defaults(func=command_polymarket_features)

    parser_polymarket_action_value = subparsers.add_parser(
        "polymarket-action-value",
        help="materialize bounded causal Polymarket execution labels",
        description=(
            "Build the frozen Round 9 BTC/ETH/SOL action-value dataset in "
            "resumable synchronized market batches. Segmented evidence is accepted "
            "only after the built-in label-free continuity audit retains at least "
            "30 post-contract synchronized groups."
        ),
    )
    parser_polymarket_action_value.add_argument(
        "--database", default="data/polymarket-paper.duckdb"
    )
    parser_polymarket_action_value.add_argument("--run-id", required=True)
    parser_polymarket_action_value.add_argument(
        "--market-groups-per-batch", type=int, default=1
    )
    parser_polymarket_action_value.add_argument("--memory-limit", default="4GB")
    parser_polymarket_action_value.add_argument(
        "--database-threads", type=int, default=1
    )
    parser_polymarket_action_value.add_argument(
        "--allow-segmented-gaps",
        action="store_true",
        help=(
            "automatically audit local market windows and use only hash-bound "
            "eligible synchronized groups"
        ),
    )
    parser_polymarket_action_value.add_argument("--json", action="store_true")
    parser_polymarket_action_value.set_defaults(
        func=command_polymarket_action_value
    )

    parser_polymarket_continuity = subparsers.add_parser(
        "polymarket-continuity",
        help="audit label-free Round 9 synchronized market continuity",
        description=(
            "Evaluate recorder errors, stream gaps, connection segments, market "
            "snapshot timing, and fresh CLOB baselines without consulting outcomes, "
            "labels, utilities, or model scores."
        ),
    )
    parser_polymarket_continuity.add_argument(
        "--database", default="data/polymarket-paper.duckdb"
    )
    parser_polymarket_continuity.add_argument("--run-id", required=True)
    parser_polymarket_continuity.add_argument("--memory-limit", default="4GB")
    parser_polymarket_continuity.add_argument(
        "--database-threads", type=int, default=1
    )
    parser_polymarket_continuity.add_argument("--json", action="store_true")
    parser_polymarket_continuity.set_defaults(func=command_polymarket_continuity)

    parser_polymarket_ridge = subparsers.add_parser(
        "polymarket-ridge",
        help="fit and audit the frozen Round 9 causal ridge baseline",
        description=(
            "Reconstruct hash-bound causal actions from one confirmation-eligible "
            "Round 9 pipeline, select the frozen ridge and threshold candidates on "
            "validation, evaluate the untouched test partition exactly once, and "
            "persist the complete audit trail. This command grants no trading or "
            "profitability authority."
        ),
    )
    parser_polymarket_ridge.add_argument(
        "--database", default="data/polymarket-paper.duckdb"
    )
    parser_polymarket_ridge.add_argument(
        "--pipeline-report-sha256",
        required=True,
        help="immutable report digest from polymarket-action-value",
    )
    parser_polymarket_ridge.add_argument("--memory-limit", default="4GB")
    parser_polymarket_ridge.add_argument("--database-threads", type=int, default=1)
    parser_polymarket_ridge.add_argument("--json", action="store_true")
    parser_polymarket_ridge.set_defaults(func=command_polymarket_ridge)

    parser_polymarket_model = subparsers.add_parser(
        "polymarket-model",
        help="run leakage-safe probability and execution research on prospective evidence",
        description=(
            "Fit a bounded residual around the Polymarket-implied probability with "
            "purged chronological BTC/ETH/SOL market groups, then compare it with "
            "the unchanged market baseline using full-resolution FOK paper replay. "
            "The resulting artifact has no live trading or profitability authority."
        ),
    )
    parser_polymarket_model.add_argument(
        "--database", default="data/polymarket-paper.duckdb"
    )
    parser_polymarket_model.add_argument("--run-id", default=None)
    parser_polymarket_model.add_argument("--cadence-ms", type=int, default=250)
    parser_polymarket_model.add_argument("--warmup-ms", type=int, default=5_000)
    parser_polymarket_model.add_argument(
        "--minimum-resolved-markets-per-asset", type=int, default=30
    )
    parser_polymarket_model.add_argument(
        "--allow-segmented-gaps",
        action="store_true",
        help=_SEGMENTED_GAPS_HELP,
    )
    parser_polymarket_model.add_argument(
        "--latency-ms",
        type=int,
        default=100,
        help="primary assumed network order latency used by causal full-depth replay",
    )
    parser_polymarket_model.add_argument(
        "--latency-stress-ms",
        default="50,100,250,500,1000",
        help="predeclared comma-separated network latencies for execution sensitivity",
    )
    parser_polymarket_model.add_argument(
        "--max-execution-observation-delay-ms",
        type=int,
        default=500,
        help=(
            "fail closed when no causal book update confirms simulated order arrival "
            "within this window"
        ),
    )
    parser_polymarket_model.add_argument(
        "--minimum-edge",
        default="0.02",
        help="minimum expected net payout per outcome contract after taker fees",
    )
    parser_polymarket_model.add_argument("--initial-capital", default="1000")
    parser_polymarket_model.add_argument(
        "--maximum-loss-fraction-per-market", default="0.005"
    )
    parser_polymarket_model.add_argument(
        "--maximum-loss-fraction-per-time-group", default="0.015"
    )
    parser_polymarket_model.add_argument(
        "--disable-ai",
        action="store_true",
        help="skip the default local multibillion-parameter veto ablation",
    )
    parser_polymarket_model.add_argument("--ai-model", default="qwen3:8b")
    parser_polymarket_model.add_argument(
        "--ai-benchmark",
        default="docs/ai/risk-review/latest/comparison.json",
        help="frozen adversarial risk benchmark that must select the requested model",
    )
    parser_polymarket_model.add_argument(
        "--ai-url", default="http://127.0.0.1:11434"
    )
    parser_polymarket_model.add_argument("--ai-timeout", type=float, default=30.0)
    parser_polymarket_model.add_argument(
        "--ai-min-confidence", type=float, default=0.65
    )
    parser_polymarket_model.add_argument(
        "--ai-max-latency-seconds", type=float, default=15.0
    )
    parser_polymarket_model.add_argument(
        "--output",
        default=None,
        help="optional deterministic JSON artifact path",
    )
    parser_polymarket_model.add_argument("--memory-limit", default="1GB")
    parser_polymarket_model.add_argument("--database-threads", type=int, default=2)
    parser_polymarket_model.add_argument("--json", action="store_true")
    parser_polymarket_model.set_defaults(func=command_polymarket_model)

    parser_polymarket_verify = subparsers.add_parser(
        "polymarket-verify",
        help="reconstruct a Polymarket model artifact from its source database",
        description=(
            "Independently rebuild features, the chronological split, deterministic "
            "model fit, held-out predictions, and every execution-latency scenario "
            "from the immutable recorder database. This command has no trading "
            "authority."
        ),
    )
    parser_polymarket_verify.add_argument("--artifact", required=True)
    parser_polymarket_verify.add_argument(
        "--database", default="data/polymarket-paper.duckdb"
    )
    parser_polymarket_verify.add_argument(
        "--output",
        default=None,
        help="optional deterministic source-verification JSON path",
    )
    parser_polymarket_verify.add_argument("--memory-limit", default="1GB")
    parser_polymarket_verify.add_argument("--database-threads", type=int, default=2)
    parser_polymarket_verify.add_argument("--json", action="store_true")
    parser_polymarket_verify.set_defaults(func=command_polymarket_verify)

    parser_polymarket_publish = subparsers.add_parser(
        "polymarket-publish",
        help="publish hash-verified Polymarket model tables and charts",
        description=(
            "Validate one prospective experiment artifact and derive every current "
            "result table, chart, report, and integrity hash from it. Publication "
            "fails closed on provenance drift or unsupported claims."
        ),
    )
    parser_polymarket_publish.add_argument("--artifact", required=True)
    parser_polymarket_publish.add_argument(
        "--database",
        default="data/polymarket-paper.duckdb",
        help="immutable recorder database independently reconstructed before publication",
    )
    parser_polymarket_publish.add_argument(
        "--research-root",
        default="docs/model-research/polymarket",
    )
    parser_polymarket_publish.add_argument("--round", type=int, default=3)
    parser_polymarket_publish.add_argument(
        "--prior-round",
        default="docs/model-research/polymarket/round-002-prospective-pipeline-evidence.json",
    )
    parser_polymarket_publish.add_argument("--memory-limit", default="1GB")
    parser_polymarket_publish.add_argument("--database-threads", type=int, default=2)
    parser_polymarket_publish.add_argument("--json", action="store_true")
    parser_polymarket_publish.set_defaults(func=command_polymarket_publish)

    parser_polymarket_paper = subparsers.add_parser(
        "polymarket-paper",
        help="inspect or execute BTC/ETH/SOL 5-minute paper orders on recorded evidence",
        description=(
            "Use the same durable ownership and reconciliation lifecycle as Binance "
            "paper trading against a validated prospective Polymarket recorder run. "
            "Strict gap-free replay is the default. This command has no authenticated "
            "or live-money order path."
        ),
    )
    parser_polymarket_paper.add_argument("--database", default="data/polymarket-paper.duckdb")
    parser_polymarket_paper.add_argument("--run-id", default=None)
    parser_polymarket_paper.add_argument(
        "--action",
        choices=[
            "status",
            "resume",
            "pause",
            "open",
            "close",
            "settle",
            "stop",
            "run-model",
        ],
        default="status",
    )
    parser_polymarket_paper.add_argument(
        "--control-path",
        default=None,
        help="optional operator-state path; defaults beside the evidence database",
    )
    parser_polymarket_paper.add_argument("--event-id", default=None)
    parser_polymarket_paper.add_argument("--position-id", default=None)
    parser_polymarket_paper.add_argument("--opening-intent-id", default=None)
    parser_polymarket_paper.add_argument("--outcome", choices=["Up", "Down"], default=None)
    parser_polymarket_paper.add_argument("--quantity", default=None)
    parser_polymarket_paper.add_argument("--limit-price", default=None)
    parser_polymarket_paper.add_argument("--latency-ms", type=int, default=None)
    parser_polymarket_paper.add_argument(
        "--artifact",
        default=None,
        help="source-verified model artifact required by --action run-model",
    )
    parser_polymarket_paper.add_argument(
        "--source-verification",
        default=None,
        help="independent source-reconstruction report required by --action run-model",
    )
    parser_polymarket_paper.add_argument(
        "--policy",
        choices=["auto", "baseline", "model", "ai"],
        default="auto",
        help="verified held-out policy used by --action run-model",
    )
    parser_polymarket_paper.add_argument(
        "--allow-unconfirmed-research",
        action="store_true",
        help=(
            "paper diagnostics only: admit an unconfirmed held-out policy while "
            "retaining all execution and stop safeguards"
        ),
    )
    parser_polymarket_paper.add_argument(
        "--output",
        default=None,
        help="optional atomic JSON report path for --action run-model",
    )
    parser_polymarket_paper.add_argument(
        "--max-execution-observation-delay-ms",
        type=int,
        default=500,
        help=(
            "fail closed when no causal book update confirms simulated order arrival "
            "within this window"
        ),
    )
    parser_polymarket_paper.add_argument(
        "--decision-delay-ms",
        type=int,
        default=0,
        help="measured model or AI review delay before order submission",
    )
    parser_polymarket_paper.add_argument(
        "--order-type",
        choices=["FAK", "FOK"],
        default="FAK",
        help="aggressive paper order fill policy",
    )
    parser_polymarket_paper.add_argument(
        "--allow-segmented-gaps",
        action="store_true",
        help=_SEGMENTED_GAPS_HELP,
    )
    parser_polymarket_paper.add_argument("--memory-limit", default="1GB")
    parser_polymarket_paper.add_argument("--database-threads", type=int, default=2)
    parser_polymarket_paper.add_argument("--json", action="store_true")
    parser_polymarket_paper.set_defaults(func=command_polymarket_paper)

    parser_archive_sync = subparsers.add_parser(
        "archive-sync",
        help="plan or ingest official Binance public archive ZIPs into SQLite",
    )
    parser_archive_sync.add_argument("--db", default="data/market_data.sqlite")
    parser_archive_sync.add_argument("--symbol", default=None)
    parser_archive_sync.add_argument("--symbols", default=None, help="comma-separated symbols; overrides --symbol")
    parser_archive_sync.add_argument("--top-symbols", type=int, default=0, help="auto-rank this many high-liquidity symbols")
    parser_archive_sync.add_argument("--quote-asset", default=None, help="quote asset used with --top-symbols")
    parser_archive_sync.add_argument("--max-scan", type=int, default=250, help="maximum universe candidates scanned with --top-symbols")
    parser_archive_sync.add_argument(
        "--min-history-months",
        type=int,
        default=0,
        help="with --top-symbols and monthly cadence, require this many monthly archive files before selecting a symbol",
    )
    parser_archive_sync.add_argument("--interval", default=None)
    parser_archive_sync.add_argument("--market", choices=["spot", "futures"], default="spot")
    parser_archive_sync.add_argument("--cadence", choices=["monthly", "daily"], default="monthly")
    parser_archive_sync.add_argument(
        "--data-type",
        choices=["klines", "aggTrades"],
        default=None,
        help="official archive data type; futures 1s defaults to aggTrades and aggregates real trades to 1s candles",
    )
    parser_archive_sync.add_argument("--max-files", type=int, default=None, help="optional safety cap for smoke runs")
    parser_archive_sync.add_argument("--start-period", default=None, help="inclusive archive period start, YYYY-MM or YYYY-MM-DD")
    parser_archive_sync.add_argument("--end-period", default=None, help="inclusive archive period end, YYYY-MM or YYYY-MM-DD")
    parser_archive_sync.add_argument("--plan-only", action="store_true", help="list the bounded archive plan without downloading files")
    parser_archive_sync.add_argument(
        "--max-planned-gb",
        type=float,
        default=50.0,
        help="block non-plan archive downloads above this planned S3 ZIP size; use 0 to disable",
    )
    parser_archive_sync.add_argument("--timeout", type=int, default=120)
    parser_archive_sync.add_argument("--force", action="store_true")
    parser_archive_sync.add_argument(
        "--aggregate-only",
        action="store_true",
        help="for aggTrades, persist derived 1s candles without duplicating raw trades",
    )
    parser_archive_sync.add_argument("--no-verify-checksum", action="store_true", help="skip Binance .CHECKSUM sidecar verification")
    parser_archive_sync.add_argument("--require-checksum", action="store_true", help="fail archive files without a readable .CHECKSUM sidecar")
    parser_archive_sync.add_argument("--json", action="store_true")
    parser_archive_sync.set_defaults(func=command_archive_sync)

    parser_microstructure = subparsers.add_parser(
        "microstructure-capture",
        help="capture real Binance futures L2, trades, and BBO data for HftBacktest replay",
    )
    parser_microstructure.add_argument(
        "--symbols",
        default=None,
        help="comma-separated supported futures symbols; defaults to configured BTC/ETH/SOL symbols",
    )
    parser_microstructure.add_argument("--seconds", type=float, default=60.0)
    parser_microstructure.add_argument("--output-root", default="data/microstructure")
    parser_microstructure.add_argument("--db", default="data/market_data.sqlite")
    parser_microstructure.add_argument("--timeout", type=float, default=10.0)
    parser_microstructure.add_argument(
        "--no-convert",
        action="store_false",
        dest="convert",
        help="capture and validate raw feeds without producing HftBacktest NPZ files",
    )
    parser_microstructure.add_argument("--json", action="store_true")
    parser_microstructure.set_defaults(convert=True, func=command_microstructure_capture)

    parser_tick_archive = subparsers.add_parser(
        "tick-archive-sync",
        help="ingest checksummed Binance futures BBO/trade archives into DuckDB",
    )
    parser_tick_archive.add_argument(
        "--symbols",
        default=None,
        help="comma-separated BTC/ETH/SOL futures symbols; defaults to runtime symbols",
    )
    parser_tick_archive.add_argument(
        "--data-types",
        default="bookTicker,trades",
        help="comma-separated official products: bookTicker,trades,bookDepth",
    )
    parser_tick_archive.add_argument("--start-date", default=None, help="inclusive UTC date, YYYY-MM-DD")
    parser_tick_archive.add_argument("--end-date", default=None, help="inclusive UTC date, YYYY-MM-DD")
    parser_tick_archive.add_argument(
        "--full-history",
        action="store_true",
        help="discover and select every official file independently for each symbol/data type",
    )
    parser_tick_archive.add_argument(
        "--available-only",
        action="store_true",
        help="record but do not fail on unavailable symbol/data-type dates",
    )
    parser_tick_archive.add_argument(
        "--plan-only",
        action="store_true",
        help="report official file coverage and compressed bytes without downloading",
    )
    parser_tick_archive.add_argument(
        "--plan-output",
        default=None,
        help="optional atomic JSON path for the compact official coverage plan",
    )
    parser_tick_archive.add_argument(
        "--max-planned-gb",
        type=float,
        default=500.0,
        help="block downloads above this official compressed-byte plan; use 0 to disable",
    )
    parser_tick_archive.add_argument("--warehouse", default="data/microstructure.duckdb")
    parser_tick_archive.add_argument("--cache-root", default="data/archive-cache")
    parser_tick_archive.add_argument("--memory-limit", default="8GB")
    parser_tick_archive.add_argument("--threads", type=int, default=8)
    parser_tick_archive.add_argument("--timeout", type=float, default=240.0)
    parser_tick_archive.add_argument("--no-retain-archive", action="store_true")
    parser_tick_archive.add_argument("--json", action="store_true")
    parser_tick_archive.set_defaults(func=command_tick_archive_sync)

    parser_tick_corpus_audit = subparsers.add_parser(
        "tick-corpus-audit",
        help="certify official inventory, manifests, and physical DuckDB partitions",
    )
    parser_tick_corpus_audit.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        help="comma-separated BTC/ETH/SOL futures symbols",
    )
    parser_tick_corpus_audit.add_argument(
        "--data-types",
        default="bookTicker,trades,bookDepth",
        help="comma-separated official products: bookTicker,trades,bookDepth",
    )
    parser_tick_corpus_audit.add_argument("--start-date", default=None)
    parser_tick_corpus_audit.add_argument("--end-date", default=None)
    parser_tick_corpus_audit.add_argument(
        "--strict-book-depth-calendar",
        action="store_false",
        dest="allow_provider_book_depth_gaps",
        help=(
            "reject dates absent from Binance's official bookDepth listing; by "
            "default those provider-proven absences are reported but permitted"
        ),
    )
    parser_tick_corpus_audit.add_argument(
        "--warehouse", default="data/microstructure.duckdb"
    )
    parser_tick_corpus_audit.add_argument("--cache-root", default="data/archive-cache")
    parser_tick_corpus_audit.add_argument("--memory-limit", default="8GB")
    parser_tick_corpus_audit.add_argument("--threads", type=int, default=8)
    parser_tick_corpus_audit.add_argument("--output", default=None)
    parser_tick_corpus_audit.add_argument("--json", action="store_true")
    parser_tick_corpus_audit.set_defaults(
        allow_provider_book_depth_gaps=True,
        func=command_tick_corpus_audit,
    )

    parser_micro_train = subparsers.add_parser(
        "microstructure-train",
        help="train a purged cost-aware L1/tape model from the tick warehouse",
    )
    parser_micro_train.add_argument("--symbol", default="BTCUSDT")
    parser_micro_train.add_argument("--warehouse", default="data/microstructure.duckdb")
    parser_micro_train.add_argument("--cache-root", default="data/archive-cache")
    parser_micro_train.add_argument("--output", default="data/microstructure-model.json")
    parser_micro_train.add_argument("--horizon-seconds", type=int, default=900)
    parser_micro_train.add_argument(
        "--decision-cadence-seconds",
        type=int,
        default=5,
        help="evaluate one decision candidate every N seconds while retaining 1s features",
    )
    parser_micro_train.add_argument("--total-latency-ms", type=int, default=750)
    parser_micro_train.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser_micro_train.add_argument(
        "--additional-slippage-bps-per-side",
        type=float,
        default=1.0,
        help=(
            "adverse execution stress charged on both entry and exit notionals "
            "in addition to taker fees (default: 1 bps per side)"
        ),
    )
    parser_micro_train.add_argument("--max-quote-age-ms", type=int, default=1000)
    parser_micro_train.add_argument(
        "--reference-order-notional-quote",
        type=float,
        default=1_000.0,
        help="reference quote-currency order size used for L1 executability labels",
    )
    parser_micro_train.add_argument(
        "--max-l1-participation",
        type=float,
        default=None,
        help="maximum share of displayed top-of-book quantity; defaults by risk profile",
    )
    parser_micro_train.add_argument("--stop-loss-bps", type=float, default=None)
    parser_micro_train.add_argument("--take-profit-bps", type=float, default=None)
    parser_micro_train.add_argument(
        "--trigger-slippage-bps",
        "--stop-slippage-bps",
        dest="trigger_slippage_bps",
        type=float,
        default=1.0,
        help="adverse exit-price adjustment after a stop/take trigger (default: 1 bps)",
    )
    parser_micro_train.add_argument(
        "--risk-level",
        choices=["conservative", "regular", "aggressive"],
        default="conservative",
    )
    parser_micro_train.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default="auto",
    )
    parser_micro_train.add_argument(
        "--minimum-promotion-days",
        type=int,
        default=240,
        help=(
            "minimum observed UTC days for exact-BBO promotion; default 240 "
            "within Binance's 320-day official BBO history"
        ),
    )
    parser_micro_train.add_argument(
        "--deployment-calibration-days",
        type=int,
        default=14,
        help="recent purged tail used only to calibrate the post-validation deployment refit",
    )
    parser_micro_train.add_argument(
        "--maximum-model-age-seconds",
        type=int,
        default=86_400,
        help="hard live-inference expiry measured from the latest labeled refit row",
    )
    terminal_mode = parser_micro_train.add_mutually_exclusive_group()
    terminal_mode.add_argument(
        "--evaluate-terminal",
        action="store_true",
        dest="evaluate_terminal",
        help="disabled compatibility flag; use hash-bound microstructure-promote",
    )
    terminal_mode.add_argument(
        "--candidate-only",
        action="store_false",
        dest="evaluate_terminal",
        help="emit a selection-stage candidate without consuming the terminal holdout (default)",
    )
    parser_micro_train.add_argument("--memory-limit", default="8GB")
    parser_micro_train.add_argument("--threads", type=int, default=8)
    parser_micro_train.add_argument("--json", action="store_true")
    parser_micro_train.set_defaults(evaluate_terminal=False, func=command_microstructure_train)

    parser_micro_prequential = subparsers.add_parser(
        "microstructure-prequential",
        help="run causal rolling-refit selection evidence before terminal evaluation",
    )
    parser_micro_prequential.add_argument(
        "--input", default="data/microstructure-model.json"
    )
    parser_micro_prequential.add_argument(
        "--warehouse", default="data/microstructure.duckdb"
    )
    parser_micro_prequential.add_argument("--cache-root", default="data/archive-cache")
    parser_micro_prequential.add_argument(
        "--output", default="data/microstructure-prequential.json"
    )
    parser_micro_prequential.add_argument(
        "--predictions", default="data/microstructure-prequential-predictions.csv"
    )
    parser_micro_prequential.add_argument(
        "--chart", default="data/microstructure-prequential.svg"
    )
    parser_micro_prequential.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default="auto",
    )
    parser_micro_prequential.add_argument("--training-window-days", type=int, default=180)
    parser_micro_prequential.add_argument("--minimum-training-days", type=int, default=60)
    parser_micro_prequential.add_argument("--calibration-days", type=int, default=14)
    parser_micro_prequential.add_argument("--policy-days", type=int, default=14)
    parser_micro_prequential.add_argument("--evaluation-block-days", type=int, default=7)
    parser_micro_prequential.add_argument("--minimum-segment-rows", type=int, default=256)
    parser_micro_prequential.add_argument("--minimum-class-rows", type=int, default=128)
    parser_micro_prequential.add_argument("--bootstrap-samples", type=int, default=2000)
    parser_micro_prequential.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="diagnostic cap; any truncated run is ineligible to pass",
    )
    parser_micro_prequential.add_argument("--memory-limit", default="8GB")
    parser_micro_prequential.add_argument("--threads", type=int, default=8)
    parser_micro_prequential.add_argument("--json", action="store_true")
    parser_micro_prequential.set_defaults(func=command_microstructure_prequential)

    parser_micro_promote = subparsers.add_parser(
        "microstructure-promote",
        help="verify prequential evidence, consume terminal replay, and create a shadow candidate",
    )
    parser_micro_promote.add_argument(
        "--input", default="data/microstructure-model.json"
    )
    parser_micro_promote.add_argument(
        "--prequential-report", default="data/microstructure-prequential.json"
    )
    parser_micro_promote.add_argument(
        "--prequential-predictions",
        default="data/microstructure-prequential-predictions.csv",
    )
    parser_micro_promote.add_argument(
        "--prequential-chart", default="data/microstructure-prequential.svg"
    )
    parser_micro_promote.add_argument(
        "--warehouse", default="data/microstructure.duckdb"
    )
    parser_micro_promote.add_argument("--cache-root", default="data/archive-cache")
    parser_micro_promote.add_argument("--output", default=None)
    parser_micro_promote.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default="auto",
    )
    parser_micro_promote.add_argument("--memory-limit", default="8GB")
    parser_micro_promote.add_argument("--threads", type=int, default=8)
    parser_micro_promote.add_argument("--json", action="store_true")
    parser_micro_promote.set_defaults(func=command_microstructure_promote)

    parser_micro_refit = subparsers.add_parser(
        "microstructure-refit",
        help="resume the source-bound deployment refit for a terminal-validated artifact",
    )
    parser_micro_refit.add_argument("--input", default="data/microstructure-model.json")
    parser_micro_refit.add_argument("--output", default=None)
    parser_micro_refit.add_argument("--warehouse", default="data/microstructure.duckdb")
    parser_micro_refit.add_argument("--cache-root", default="data/archive-cache")
    parser_micro_refit.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default="auto",
    )
    parser_micro_refit.add_argument("--memory-limit", default="8GB")
    parser_micro_refit.add_argument("--threads", type=int, default=8)
    parser_micro_refit.add_argument("--json", action="store_true")
    parser_micro_refit.set_defaults(func=command_microstructure_refit)

    parser_micro_shadow = subparsers.add_parser(
        "microstructure-shadow",
        help="run the locked no-order public-feed shadow gate for a deployment refit",
    )
    parser_micro_shadow.add_argument("--input", default="data/microstructure-model.json")
    parser_micro_shadow.add_argument("--output", default=None)
    parser_micro_shadow.add_argument(
        "--seconds",
        type=float,
        default=(21_600.0 + MICROSTRUCTURE_STREAM_WARMUP_SECONDS + 60.0),
        help=(
            "public-feed capture duration; promotion requires feature warmup plus "
            "six complete evaluated hours"
        ),
    )
    parser_micro_shadow.add_argument(
        "--output-root",
        default="data/microstructure-shadow/captures",
    )
    parser_micro_shadow.add_argument(
        "--report",
        default="data/microstructure-shadow/report.json",
    )
    parser_micro_shadow.add_argument(
        "--trades",
        default="data/microstructure-shadow/trades.csv",
    )
    parser_micro_shadow.add_argument("--db", default="data/market_data.sqlite")
    parser_micro_shadow.add_argument("--timeout", type=float, default=10.0)
    parser_micro_shadow.add_argument("--json", action="store_true")
    parser_micro_shadow.set_defaults(func=command_microstructure_shadow)

    parser_tape_depth_train = subparsers.add_parser(
        "tape-depth-train",
        help="train a purged research-only gross forecaster from real trade/depth history",
    )
    parser_tape_depth_train.add_argument("--symbol", default="BTCUSDT")
    parser_tape_depth_train.add_argument(
        "--warehouse", default="data/microstructure.duckdb"
    )
    parser_tape_depth_train.add_argument("--cache-root", default="data/archive-cache")
    parser_tape_depth_train.add_argument(
        "--output", default="data/tape-depth-model.json"
    )
    parser_tape_depth_train.add_argument("--window-days", type=int, default=180)
    parser_tape_depth_train.add_argument(
        "--end-date",
        default=None,
        help="optional inclusive UTC evaluation date; defaults to latest covered target",
    )
    parser_tape_depth_train.add_argument("--horizon-seconds", type=int, default=60)
    parser_tape_depth_train.add_argument("--total-latency-ms", type=int, default=750)
    parser_tape_depth_train.add_argument(
        "--decision-cadence-seconds", type=int, default=5
    )
    parser_tape_depth_train.add_argument(
        "--maximum-depth-age-ms", type=int, default=60_000
    )
    parser_tape_depth_train.add_argument(
        "--risk-level",
        choices=["conservative", "regular", "aggressive"],
        default="conservative",
    )
    parser_tape_depth_train.add_argument(
        "--model-profile",
        choices=["regularized", "balanced", "expressive"],
        default="regularized",
    )
    parser_tape_depth_train.add_argument(
        "--feature-set",
        choices=["core", "tape_derived", "cross_asset", "full"],
        default="full",
    )
    parser_tape_depth_train.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default="auto",
    )
    parser_tape_depth_train.add_argument("--minimum-segment-rows", type=int, default=2_000)
    parser_tape_depth_train.add_argument("--maximum-rows", type=int, default=5_000_000)
    parser_tape_depth_train.add_argument("--memory-limit", default="8GB")
    parser_tape_depth_train.add_argument("--threads", type=int, default=8)
    parser_tape_depth_train.add_argument("--json", action="store_true")
    parser_tape_depth_train.set_defaults(func=command_tape_depth_train)

    parser_tape_depth_design = subparsers.add_parser(
        "tape-depth-design",
        help="write a precommitted multi-dimensional tape/depth experiment design",
    )
    parser_tape_depth_design.add_argument(
        "--risk-level",
        choices=["conservative", "regular", "aggressive"],
        default="conservative",
    )
    parser_tape_depth_design.add_argument("--sampled-count", type=int, default=24)
    parser_tape_depth_design.add_argument("--seed", type=int, default=20_260_710)
    parser_tape_depth_design.add_argument(
        "--output",
        default="data/tape-depth-experiment-design.json",
    )
    parser_tape_depth_design.add_argument("--json", action="store_true")
    parser_tape_depth_design.set_defaults(func=command_tape_depth_design)

    parser_tape_depth_study = subparsers.add_parser(
        "tape-depth-study",
        help="run and checkpoint every candidate in a precommitted screening design",
    )
    parser_tape_depth_study.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT"
    )
    parser_tape_depth_study.add_argument("--design", required=True)
    parser_tape_depth_study.add_argument(
        "--warehouse", default="data/microstructure.duckdb"
    )
    parser_tape_depth_study.add_argument("--cache-root", default="data/archive-cache")
    parser_tape_depth_study.add_argument(
        "--output-dir", default="data/tape-depth-study"
    )
    parser_tape_depth_study.add_argument("--training-window-days", type=int, default=730)
    parser_tape_depth_study.add_argument("--tuning-window-days", type=int, default=30)
    parser_tape_depth_study.add_argument(
        "--calibration-window-days", type=int, default=30
    )
    parser_tape_depth_study.add_argument(
        "--evaluation-window-days", type=int, default=90
    )
    parser_tape_depth_study.add_argument("--total-latency-ms", type=int, default=750)
    parser_tape_depth_study.add_argument("--maximum-rows", type=int, default=5_000_000)
    parser_tape_depth_study.add_argument(
        "--maximum-cached-rows", type=int, default=15_000_000
    )
    parser_tape_depth_study.add_argument(
        "--no-dataset-cache",
        action="store_false",
        dest="dataset_cache",
    )
    parser_tape_depth_study.add_argument(
        "--max-folds", type=int, choices=[4, 6, 8, 10], default=4
    )
    parser_tape_depth_study.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default="auto",
    )
    parser_tape_depth_study.add_argument(
        "--minimum-segment-rows", type=int, default=10_000
    )
    parser_tape_depth_study.add_argument("--memory-limit", default="8GB")
    parser_tape_depth_study.add_argument("--threads", type=int, default=8)
    parser_tape_depth_study.add_argument("--resume", action="store_true")
    parser_tape_depth_study.add_argument("--plan-only", action="store_true")
    parser_tape_depth_study.add_argument("--json", action="store_true")
    parser_tape_depth_study.set_defaults(
        dataset_cache=True,
        func=command_tape_depth_study,
    )

    parser_tape_depth_prequential = subparsers.add_parser(
        "tape-depth-prequential",
        help="run timestamp-defined rolling gross-forecast evidence across major symbols",
    )
    parser_tape_depth_prequential.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT"
    )
    parser_tape_depth_prequential.add_argument(
        "--warehouse", default="data/microstructure.duckdb"
    )
    parser_tape_depth_prequential.add_argument(
        "--cache-root", default="data/archive-cache"
    )
    parser_tape_depth_prequential.add_argument(
        "--output-dir", default="data/tape-depth-prequential"
    )
    parser_tape_depth_prequential.add_argument(
        "--training-window-days", type=int, default=730
    )
    parser_tape_depth_prequential.add_argument(
        "--tuning-window-days", type=int, default=30
    )
    parser_tape_depth_prequential.add_argument(
        "--calibration-window-days", type=int, default=30
    )
    parser_tape_depth_prequential.add_argument(
        "--evaluation-window-days", type=int, default=90
    )
    parser_tape_depth_prequential.add_argument(
        "--horizon-seconds",
        type=int,
        default=None,
        help="default 60; sealed confirmation derives the frozen winner",
    )
    parser_tape_depth_prequential.add_argument("--total-latency-ms", type=int, default=750)
    parser_tape_depth_prequential.add_argument(
        "--decision-cadence-seconds",
        type=int,
        default=None,
        help="default 20; sealed confirmation derives the frozen winner",
    )
    parser_tape_depth_prequential.add_argument(
        "--maximum-depth-age-ms",
        type=int,
        default=None,
        help="default 60000; sealed confirmation derives the frozen winner",
    )
    parser_tape_depth_prequential.add_argument("--maximum-rows", type=int, default=5_000_000)
    parser_tape_depth_prequential.add_argument(
        "--maximum-cached-rows", type=int, default=15_000_000
    )
    parser_tape_depth_prequential.add_argument(
        "--no-dataset-cache",
        action="store_false",
        dest="dataset_cache",
        help="disable the verified DuckDB derived-dataset cache",
    )
    parser_tape_depth_prequential.add_argument(
        "--study-stage",
        choices=["development", "screening", "confirmation"],
        default="development",
    )
    parser_tape_depth_prequential.add_argument(
        "--selection-lock",
        help="winner lock required for sealed confirmation",
    )
    parser_tape_depth_prequential.add_argument("--max-folds", type=int, default=0)
    parser_tape_depth_prequential.add_argument(
        "--risk-level",
        choices=["conservative", "regular", "aggressive"],
        default="conservative",
    )
    parser_tape_depth_prequential.add_argument(
        "--model-profile",
        choices=["regularized", "balanced", "expressive"],
        default=None,
        help="default regularized; confirmation derives the frozen winner",
    )
    parser_tape_depth_prequential.add_argument(
        "--feature-set",
        choices=["core", "tape_derived", "cross_asset", "full"],
        default=None,
        help="default full; confirmation derives the frozen winner",
    )
    parser_tape_depth_prequential.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default="auto",
    )
    parser_tape_depth_prequential.add_argument(
        "--minimum-segment-rows", type=int, default=10_000
    )
    parser_tape_depth_prequential.add_argument("--memory-limit", default="8GB")
    parser_tape_depth_prequential.add_argument("--threads", type=int, default=8)
    parser_tape_depth_prequential.add_argument("--plan-only", action="store_true")
    parser_tape_depth_prequential.add_argument("--resume", action="store_true")
    parser_tape_depth_prequential.add_argument("--json", action="store_true")
    parser_tape_depth_prequential.set_defaults(
        dataset_cache=True,
        func=command_tape_depth_prequential,
    )

    parser_tape_depth_select = subparsers.add_parser(
        "tape-depth-select",
        help="freeze one winner from screening-only tape/depth reports",
    )
    parser_tape_depth_select.add_argument(
        "--report",
        action="append",
        required=True,
        help="screening report path; repeat for every declared trial",
    )
    parser_tape_depth_select.add_argument(
        "--design",
        required=True,
        help="precommitted multi-fidelity experiment design JSON",
    )
    parser_tape_depth_select.add_argument(
        "--output", default="data/tape-depth-selection.json"
    )
    parser_tape_depth_select.add_argument("--json", action="store_true")
    parser_tape_depth_select.set_defaults(func=command_tape_depth_select)

    parser_tape_depth_confirm = subparsers.add_parser(
        "tape-depth-confirm",
        help="confirm only the frozen tape/depth winner on untouched folds",
    )
    parser_tape_depth_confirm.add_argument("--selection", required=True)
    parser_tape_depth_confirm.add_argument("--report", required=True)
    parser_tape_depth_confirm.add_argument(
        "--output", default="data/tape-depth-confirmation.json"
    )
    parser_tape_depth_confirm.add_argument("--json", action="store_true")
    parser_tape_depth_confirm.set_defaults(func=command_tape_depth_confirm)

    parser_tape_depth_execution_confirm = subparsers.add_parser(
        "tape-depth-execution-confirm",
        help="run the frozen exact-BBO after-cost confirmation dates",
    )
    parser_tape_depth_execution_confirm.add_argument(
        "--design",
        default="docs/model-research/tape-depth/confirmation-design.json",
    )
    parser_tape_depth_execution_confirm.add_argument(
        "--availability",
        default="docs/microstructure/availability.json",
    )
    parser_tape_depth_execution_confirm.add_argument(
        "--warehouse", default="data/microstructure.duckdb"
    )
    parser_tape_depth_execution_confirm.add_argument(
        "--cache-root", default="data/archive-cache"
    )
    parser_tape_depth_execution_confirm.add_argument(
        "--output-dir", default="data/tape-depth-execution-confirmation"
    )
    parser_tape_depth_execution_confirm.add_argument("--memory-limit", default="8GB")
    parser_tape_depth_execution_confirm.add_argument("--threads", type=int, default=8)
    parser_tape_depth_execution_confirm.add_argument("--resume", action="store_true")
    parser_tape_depth_execution_confirm.add_argument("--json", action="store_true")
    parser_tape_depth_execution_confirm.set_defaults(
        func=command_tape_depth_execution_confirm
    )

    parser_train = subparsers.add_parser("train", help="train model from cached candles")
    parser_train.add_argument("--input", default="data/historical_market.json")
    parser_train.add_argument("--output", default="data/model.json")
    parser_train.add_argument("--source", choices=["auto", "file", "db"], default="auto")
    parser_train.add_argument("--db", default="data/market_data.sqlite")
    parser_train.add_argument("--interval", default=None)
    parser_train.add_argument("--market", choices=["spot", "futures"], default=None)
    parser_train.add_argument("--min-rows", type=int, default=120)
    parser_train.add_argument("--download-missing", action="store_true")
    parser_train.add_argument("--preset", choices=sorted(_TRAINING_PRESETS), default="custom")
    parser_train.add_argument("--epochs", type=int, default=250)
    parser_train.add_argument("--learning-rate", type=float, default=0.05)
    parser_train.add_argument("--l2-penalty", type=float, default=1e-4)
    parser_train.add_argument("--seed", type=int, default=7)
    parser_train.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="training backend override; default uses saved runtime compute_backend",
    )
    parser_train.add_argument("--batch-size", type=int, default=8192, help="mini-batch size for GPU training")
    parser_train.add_argument("--walk-forward", action="store_true", help="run walk-forward validation before final training")
    parser_train.add_argument("--walk-forward-train", type=int, default=300)
    parser_train.add_argument("--walk-forward-test", type=int, default=60)
    parser_train.add_argument("--walk-forward-step", type=int, default=30)
    parser_train.add_argument("--calibrate-threshold", action="store_true", help="optimize a probability threshold on validation split")
    parser_train.set_defaults(func=command_train)

    parser_prepare = subparsers.add_parser("prepare", help="fetch, train, evaluate, backtest, then run readiness checks")
    parser_prepare.add_argument("--historical", default="data/historical_market.json")
    parser_prepare.add_argument("--model", default="data/model.json")
    parser_prepare.add_argument("--limit", type=int, default=500)
    parser_prepare.add_argument("--batch-size", type=int, default=1000, help="klines per fetch request (spot max 1000, futures max 1500)")
    parser_prepare.add_argument("--preset", choices=sorted(_TRAINING_PRESETS), default="balanced")
    parser_prepare.add_argument("--epochs", type=int, default=None, help="override preset training epochs")
    parser_prepare.add_argument("--learning-rate", type=float, default=0.05)
    parser_prepare.add_argument("--l2-penalty", type=float, default=1e-4)
    parser_prepare.add_argument("--seed", type=int, default=7)
    parser_prepare.add_argument("--start-cash", type=float, default=1000.0)
    parser_prepare.add_argument("--walk-forward", action="store_true", dest="walk_forward", help="force walk-forward validation")
    parser_prepare.add_argument("--no-walk-forward", action="store_false", dest="walk_forward", help="skip walk-forward validation")
    parser_prepare.add_argument("--walk-forward-train", type=int, default=None, help="override walk-forward training window")
    parser_prepare.add_argument("--walk-forward-test", type=int, default=None, help="override walk-forward test window")
    parser_prepare.add_argument("--walk-forward-step", type=int, default=None, help="override walk-forward step")
    parser_prepare.add_argument("--calibrate-threshold", action="store_true", dest="calibrate_threshold", help="force threshold calibration")
    parser_prepare.add_argument("--no-calibrate-threshold", action="store_false", dest="calibrate_threshold", help="skip threshold calibration")
    parser_prepare.set_defaults(walk_forward=None, calibrate_threshold=None)
    parser_prepare.add_argument("--online-doctor", action="store_true", help="include exchange connectivity in final readiness checks")
    parser_prepare.set_defaults(func=command_prepare)

    parser_tune = subparsers.add_parser("tune", help="perform a focused walk-forward tune over few risk parameters")
    parser_tune.add_argument("--input", default="data/historical_market.json")
    parser_tune.add_argument("--save-best", action="store_true")
    parser_tune.add_argument("--min-risk", type=float, default=0.002)
    parser_tune.add_argument("--max-risk", type=float, default=0.02)
    parser_tune.add_argument("--steps", type=int, default=5)
    parser_tune.add_argument("--min-leverage", type=float, default=1.0)
    parser_tune.add_argument("--max-leverage", type=float, default=20.0)
    parser_tune.add_argument("--min-threshold", type=float, default=0.52)
    parser_tune.add_argument("--max-threshold", type=float, default=0.88)
    parser_tune.add_argument("--min-take", type=float, default=0.01)
    parser_tune.add_argument("--max-take", type=float, default=0.06)
    parser_tune.add_argument("--min-stop", type=float, default=0.008)
    parser_tune.add_argument("--max-stop", type=float, default=0.04)
    parser_tune.add_argument("--compute-backend", choices=_COMPUTE_BACKEND_CHOICES, default=None)
    parser_tune.add_argument("--batch-size", type=int, default=8192, help="mini-batch size for accelerated tuning")
    parser_tune.add_argument("--lookback-days", type=int, default=None, help="use only the most recent N days of candles for tuning")
    parser_tune.add_argument("--from-date", default=None, help="inclusive start date for tuning window (YYYY-MM-DD)")
    parser_tune.add_argument("--to-date", default=None, help="inclusive end date for tuning window (YYYY-MM-DD)")
    parser_tune.set_defaults(func=command_tune)

    parser_backtest = subparsers.add_parser("backtest", help="run backtest against cached data")
    parser_backtest.add_argument("--input", default="data/historical_market.json")
    parser_backtest.add_argument("--model", default="data/model.json")
    parser_backtest.add_argument("--start-cash", type=float, default=1000.0)
    parser_backtest.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="model-scoring backend override; default uses saved runtime compute_backend",
    )
    parser_backtest.add_argument(
        "--score-batch-size",
        type=int,
        default=8192,
        help="batch size for GPU-assisted probability scoring",
    )
    parser_backtest.add_argument(
        "--execution-db",
        default=None,
        help="optional SQLite market-data DB; latest typed top-of-book row becomes symbol-specific fill stress",
    )
    parser_backtest.set_defaults(func=command_backtest)

    parser_backtest_chart = subparsers.add_parser("backtest-chart", help="run backtest and save an SVG performance chart")
    parser_backtest_chart.add_argument("--input", default="data/historical_market.json")
    parser_backtest_chart.add_argument("--model", default="data/model.json")
    parser_backtest_chart.add_argument("--output", default="data/backtest_performance.svg")
    parser_backtest_chart.add_argument("--start-cash", type=float, default=1000.0)
    parser_backtest_chart.add_argument("--compute-backend", choices=_COMPUTE_BACKEND_CHOICES, default=None)
    parser_backtest_chart.add_argument("--score-batch-size", type=int, default=8192)
    parser_backtest_chart.add_argument(
        "--execution-db",
        default=None,
        help="optional SQLite market-data DB for symbol-specific top-of-book fill stress",
    )
    parser_backtest_chart.set_defaults(func=command_backtest_chart)

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate saved model against cached candles")
    parser_evaluate.add_argument("--input", default="data/historical_market.json")
    parser_evaluate.add_argument("--model", default="data/model.json")
    parser_evaluate.add_argument("--threshold", type=float, default=None)
    parser_evaluate.add_argument("--calibrate-threshold", action="store_true")
    parser_evaluate.set_defaults(func=command_evaluate)

    parser_signals = subparsers.add_parser(
        "signals",
        help="fetch and cache free external market signal checks used by live mode",
    )
    parser_signals.add_argument("--model", default="data/model.json", help="model path used to derive default cache location")
    parser_signals.add_argument("--cache", default=None, help="signal cache path (default: model-adjacent data/signals)")
    parser_signals.add_argument("--ttl", type=int, default=300, help="cache TTL seconds")
    parser_signals.add_argument("--timeout", type=float, default=3.0, help="per-provider timeout seconds")
    parser_signals.add_argument("--max-adjustment", type=float, default=0.04, help="maximum model score adjustment")
    parser_signals.add_argument("--min-providers", type=int, default=2, help="minimum usable providers for positive boosts")
    parser_signals.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="optional backend for news keyword scoring",
    )
    parser_signals.add_argument(
        "--short-reaction-refresh",
        type=int,
        default=30,
        help="seconds after which cached short-horizon reaction news must refresh",
    )
    parser_signals.add_argument("--news-provider-limit", type=int, default=None, help="maximum RSS/news providers to poll")
    parser_signals.add_argument("--news-items-per-provider", type=int, default=None, help="feed items scored per news provider")
    parser_signals.add_argument("--provider-parallelism", type=int, default=None, help="maximum simultaneous news provider requests")
    parser_signals.add_argument("--provider-jitter", type=float, default=None, help="random per-provider delay ceiling in seconds")
    parser_signals.add_argument("--ollama-news", action="store_true", default=None, help="enable Ollama AI headline evaluation")
    parser_signals.add_argument("--no-ollama-news", action="store_false", dest="ollama_news", help="disable Ollama AI headline evaluation")
    parser_signals.add_argument("--ollama-model", default=None)
    parser_signals.add_argument("--ollama-url", default=None)
    parser_signals.add_argument("--ollama-timeout", type=float, default=None)
    parser_signals.add_argument("--telemetry-db", default=None, help="SQLite raw telemetry DB path")
    parser_signals.add_argument("--source-grade-max-age-hours", type=float, default=None, help="ignore source grades older than this; 0 disables the age cap")
    parser_signals.add_argument("--no-telemetry", action="store_true", help="do not journal raw provider/model payloads")
    parser_signals.add_argument("--loop", action="store_true", help="poll repeatedly with jitter instead of one collection")
    parser_signals.add_argument("--iterations", type=int, default=0, help="loop iterations; 0 means until interrupted")
    parser_signals.add_argument("--sleep", type=float, default=None, help="base loop interval seconds")
    parser_signals.add_argument("--jitter", type=float, default=None, help="random loop delay ceiling in seconds")
    parser_signals.add_argument("--refresh", action="store_true", help="ignore cache and fetch every provider")
    parser_signals.add_argument("--json", action="store_true", help="print machine-readable report")
    parser_signals.set_defaults(func=command_signals)

    parser_signals_benchmark = subparsers.add_parser(
        "signals-benchmark",
        help="benchmark provider polling limits, parallelism, and optional Ollama latency",
    )
    parser_signals_benchmark.add_argument("--provider-limit", action="append", type=int, default=None)
    parser_signals_benchmark.add_argument("--parallelism", action="append", type=int, default=None)
    parser_signals_benchmark.add_argument("--iterations", type=int, default=1)
    parser_signals_benchmark.add_argument("--timeout", type=float, default=3.0)
    parser_signals_benchmark.add_argument("--provider-jitter", type=float, default=0.0)
    parser_signals_benchmark.add_argument("--ollama-news", action="store_true", default=None)
    parser_signals_benchmark.add_argument("--no-ollama-news", action="store_false", dest="ollama_news")
    parser_signals_benchmark.add_argument("--ollama-model", default=None)
    parser_signals_benchmark.add_argument("--ollama-url", default=None)
    parser_signals_benchmark.add_argument("--ollama-timeout", type=float, default=None)
    parser_signals_benchmark.add_argument("--cache", default="data/signals/benchmark_external_signals.json")
    parser_signals_benchmark.add_argument("--no-telemetry", action="store_true")
    parser_signals_benchmark.add_argument("--json", action="store_true")
    parser_signals_benchmark.set_defaults(func=command_signals_benchmark)

    parser_source_grades = subparsers.add_parser(
        "source-grades",
        help="grade raw signal/news/model sources from telemetry with optional Ollama review",
    )
    parser_source_grades.add_argument("--db", default=None, help="SQLite raw telemetry DB path")
    parser_source_grades.add_argument("--window-hours", type=float, default=None)
    parser_source_grades.add_argument("--ollama", action="store_true", default=None, help="enable Ollama grading")
    parser_source_grades.add_argument("--no-ollama", action="store_false", dest="ollama", help="disable Ollama grading")
    parser_source_grades.add_argument("--ollama-model", default=None)
    parser_source_grades.add_argument("--ollama-url", default=None)
    parser_source_grades.add_argument("--ollama-timeout", type=float, default=None)
    parser_source_grades.add_argument("--json", action="store_true")
    parser_source_grades.set_defaults(func=command_source_grades)

    parser_live = subparsers.add_parser("live", help="run a conservative live loop on testnet/demo or paper mode")
    parser_live.add_argument("--model", default="data/model.json")
    parser_live.add_argument("--steps", type=int, default=20)
    parser_live.add_argument("--sleep", type=int, default=5)
    parser_live.add_argument("--leverage", type=float, default=None, help="override leverage for this run (futures only)")
    parser_live.add_argument(
        "--retrain-interval",
        type=int,
        default=0,
        help="retrain model every N steps (0 disables, for adaptive paper/live behavior)",
    )
    parser_live.add_argument(
        "--retrain-window",
        type=int,
        default=300,
        help="number of recent rows used for each live retrain",
    )
    parser_live.add_argument(
        "--retrain-min-rows",
        type=int,
        default=240,
        help="minimum rows required before a retrain is attempted",
    )
    parser_live.add_argument("--compute-backend", choices=_COMPUTE_BACKEND_CHOICES, default=None)
    parser_live.add_argument("--batch-size", type=int, default=8192, help="mini-batch size for live retraining")
    parser_live.add_argument(
        "--paper",
        action="store_true",
        help="force paper mode for this run even when runtime.dry_run is false",
    )
    parser_live.add_argument(
        "--live",
        action="store_true",
        help="force authenticated testnet execution even when runtime.dry_run is true",
    )
    parser_live.add_argument(
        "--external-signals",
        action="store_true",
        default=None,
        help="enable cached free external signal adjustment for this run",
    )
    parser_live.add_argument(
        "--no-external-signals",
        action="store_false",
        dest="external_signals",
        help="disable cached free external signal adjustment for this run",
    )
    parser_live.set_defaults(func=command_live)

    parser_status = subparsers.add_parser("status", help="show persisted runtime and strategy config")
    parser_status.add_argument("--compact", action="store_true", help="print one secret-free operator status line")
    parser_status.set_defaults(func=command_status)

    parser_compute = subparsers.add_parser("compute", help="show or set the model-training compute backend")
    parser_compute.add_argument("--backend", choices=_COMPUTE_BACKEND_CHOICES, default=None)
    parser_compute.set_defaults(func=command_compute)

    parser_ai = subparsers.add_parser("ai", help="show or configure local GPU AI acceleration preflight")
    parser_ai.add_argument("--enable", action="store_true", default=None, help="enable AI decision features")
    parser_ai.add_argument("--disable", action="store_true", default=None, help="disable AI decision features")
    parser_ai.add_argument("--provider", default=None, help="AI provider: auto, local-gpu, ollama, openai-compatible, etc.")
    parser_ai.add_argument("--model", default=None, help="AI model identifier or 'auto'")
    parser_ai.add_argument("--require-gpu", action="store_true", default=None)
    parser_ai.add_argument("--no-require-gpu", action="store_true", default=None)
    parser_ai.add_argument("--min-free-vram-gb", type=float, default=None)
    parser_ai.add_argument("--min-free-ram-gb", type=float, default=None)
    parser_ai.add_argument("--min-model-parameters-b", type=float, default=None)
    parser_ai.add_argument("--allow-paper-fallback", action="store_true", default=None)
    parser_ai.add_argument("--no-paper-fallback", action="store_true", default=None)
    parser_ai.add_argument("--json", action="store_true")
    parser_ai.set_defaults(func=command_ai)

    parser_ai_benchmark = subparsers.add_parser(
        "ai-benchmark",
        help="compare local AI models on structured finance-risk review cases",
    )
    parser_ai_benchmark.add_argument("--models", default="", help="comma-separated Ollama model names; defaults to installed curated candidates")
    parser_ai_benchmark.add_argument("--url", default="http://127.0.0.1:11434")
    parser_ai_benchmark.add_argument("--timeout", type=float, default=20.0)
    parser_ai_benchmark.add_argument("--minimum-score", type=float, default=0.78)
    parser_ai_benchmark.add_argument("--output", default="data/ai_model_benchmark.json")
    parser_ai_benchmark.add_argument("--json", action="store_true")
    parser_ai_benchmark.set_defaults(func=command_ai_benchmark)

    parser_ai_forecast = subparsers.add_parser(
        "ai-forecast-benchmark",
        help="benchmark a pinned financial foundation forecast on real post-cutoff BTC/ETH/SOL data",
    )
    parser_ai_forecast.add_argument("--database", default="data/market_data.sqlite")
    parser_ai_forecast.add_argument("--model-size", choices=("small", "base"), default="base")
    parser_ai_forecast.add_argument("--backend", choices=_COMPUTE_BACKEND_CHOICES, default="directml")
    parser_ai_forecast.add_argument("--source-cache", default=None)
    parser_ai_forecast.add_argument("--bootstrap-source", action="store_true")
    parser_ai_forecast.add_argument("--repair-source", action="store_true")
    parser_ai_forecast.add_argument("--allow-cpu", action="store_true")
    parser_ai_forecast.add_argument("--start", default="2024-07-01T00:00:00Z")
    parser_ai_forecast.add_argument("--end-exclusive", default="2026-01-01T00:00:00Z")
    parser_ai_forecast.add_argument("--samples-per-symbol", type=int, default=128)
    parser_ai_forecast.add_argument("--lookback-bars", type=int, default=480)
    parser_ai_forecast.add_argument("--prediction-bars", type=int, default=12)
    parser_ai_forecast.add_argument("--batch-size", type=int, default=3)
    parser_ai_forecast.add_argument("--inference-samples", type=int, default=10)
    parser_ai_forecast.add_argument("--temperature", type=float, default=0.6)
    parser_ai_forecast.add_argument("--top-k", type=int, default=0)
    parser_ai_forecast.add_argument("--top-p", type=float, default=0.9)
    parser_ai_forecast.add_argument(
        "--include-volume",
        action="store_true",
        help="include volume/amount despite the upstream crypto evaluation using OHLC only",
    )
    parser_ai_forecast.add_argument("--seed", type=int, default=17)
    parser_ai_forecast.add_argument("--bootstrap-samples", type=int, default=2000)
    parser_ai_forecast.add_argument("--worker-timeout", type=float, default=60.0)
    parser_ai_forecast.add_argument("--max-worker-restarts", type=int, default=5)
    parser_ai_forecast.add_argument("--worker-rotation-batches", type=int, default=20)
    parser_ai_forecast.add_argument(
        "--observations",
        default="data/foundation_ai/kronos_observations.csv",
    )
    parser_ai_forecast.add_argument(
        "--output",
        default="data/foundation_ai/kronos_benchmark.json",
    )
    parser_ai_forecast.add_argument(
        "--chart",
        default="data/foundation_ai/kronos_benchmark.svg",
    )
    parser_ai_forecast.add_argument("--json", action="store_true")
    parser_ai_forecast.set_defaults(func=command_ai_forecast_benchmark)

    parser_ai_review = subparsers.add_parser(
        "ai-review",
        help="run a structured local-AI risk review over a model-lab report",
    )
    parser_ai_review.add_argument("--report", default="data/model_lab/model_lab_report.json")
    parser_ai_review.add_argument("--output", default=None)
    parser_ai_review.add_argument("--model", default=None)
    parser_ai_review.add_argument("--url", default="http://127.0.0.1:11434")
    parser_ai_review.add_argument("--timeout", type=float, default=20.0)
    parser_ai_review.add_argument("--json", action="store_true")
    parser_ai_review.set_defaults(func=command_ai_review)

    parser_strategy = subparsers.add_parser("strategy", help="adjust strategy and risk parameters")
    parser_strategy.add_argument("--profile", choices=sorted(_STRATEGY_PROFILES), default="custom")
    parser_strategy.add_argument("--risk-level", choices=["conservative", "regular", "aggressive"], default=None)
    parser_strategy.add_argument("--reinvest-profits", action="store_true", default=None)
    parser_strategy.add_argument("--no-reinvest-profits", action="store_true", default=None)
    parser_strategy.add_argument("--leverage", type=float, default=None)
    parser_strategy.add_argument("--risk", type=float, default=None)
    parser_strategy.add_argument("--max-position", type=float, default=None)
    parser_strategy.add_argument("--stop", type=float, default=None)
    parser_strategy.add_argument("--take", type=float, default=None)
    parser_strategy.add_argument("--cooldown", type=int, default=None)
    parser_strategy.add_argument("--min-position-hold-bars", type=int, default=None)
    parser_strategy.add_argument("--flat-signal-exit-grace-bars", type=int, default=None)
    parser_strategy.add_argument("--max-position-hold-bars", type=int, default=None)
    parser_strategy.add_argument("--max-open", type=int, default=None)
    parser_strategy.add_argument("--min-diversified-assets", type=int, default=None)
    parser_strategy.add_argument("--max-asset-allocation", type=float, default=None)
    parser_strategy.add_argument("--max-portfolio-risk", type=float, default=None)
    parser_strategy.add_argument("--min-quote-volume-usdc", type=float, default=None)
    parser_strategy.add_argument("--min-trade-count-24h", type=int, default=None)
    parser_strategy.add_argument("--max-spread-bps", type=float, default=None)
    parser_strategy.add_argument("--min-liquidity-score", type=float, default=None)
    parser_strategy.add_argument("--unpredictability-cooldown", type=int, default=None)
    parser_strategy.add_argument("--max-regime-unpredictability", type=float, default=None)
    parser_strategy.add_argument("--max-prediction-entropy", type=float, default=None)
    parser_strategy.add_argument("--min-model-confidence", type=float, default=None)
    parser_strategy.add_argument("--max-trades-per-day", type=int, default=None)
    parser_strategy.add_argument("--signal-threshold", type=float, default=None)
    parser_strategy.add_argument("--max-drawdown", type=float, default=None)
    parser_strategy.add_argument("--max-daily-loss", type=float, default=None)
    parser_strategy.add_argument("--max-session-loss", type=float, default=None)
    parser_strategy.add_argument("--max-consecutive-losses", type=int, default=None)
    parser_strategy.add_argument("--max-network-errors", type=int, default=None)
    parser_strategy.add_argument("--recovery-cooldown-seconds", type=int, default=None)
    parser_strategy.add_argument("--taker-fee-bps", type=float, default=None)
    parser_strategy.add_argument("--slippage-bps", type=float, default=None)
    parser_strategy.add_argument("--label-threshold", type=float, default=None)
    parser_strategy.add_argument("--model-lookback", type=int, default=None)
    parser_strategy.add_argument("--training-epochs", type=int, default=None)
    parser_strategy.add_argument("--confidence-beta", type=float, default=None)
    parser_strategy.add_argument("--feature-window-short", type=int, default=None)
    parser_strategy.add_argument("--feature-window-long", type=int, default=None)
    parser_strategy.add_argument("--set-features", default=None, help="comma-separated ordered feature list for retraining")
    parser_strategy.add_argument("--enable-feature", action="append", default=None, help="enable a feature by name")
    parser_strategy.add_argument("--disable-feature", action="append", default=None, help="disable a feature by name")
    parser_strategy.add_argument("--external-signals", action="store_true", default=None, help="enable live free external signals")
    parser_strategy.add_argument("--no-external-signals", action="store_false", dest="external_signals", help="disable live free external signals")
    parser_strategy.add_argument("--external-signal-max-adjustment", type=float, default=None)
    parser_strategy.add_argument("--external-signal-min-providers", type=int, default=None)
    parser_strategy.add_argument("--external-signal-ttl", type=int, default=None)
    parser_strategy.add_argument("--external-signal-timeout", type=float, default=None)
    parser_strategy.add_argument("--external-news-ai", action="store_true", default=None)
    parser_strategy.add_argument("--no-external-news-ai", action="store_false", dest="external_news_ai")
    parser_strategy.add_argument("--external-news-ai-model", default=None)
    parser_strategy.add_argument("--external-news-ai-url", default=None)
    parser_strategy.add_argument("--external-news-ai-timeout", type=float, default=None)
    parser_strategy.add_argument("--external-news-provider-limit", type=int, default=None)
    parser_strategy.add_argument("--external-provider-parallelism", type=int, default=None)
    parser_strategy.add_argument("--external-provider-jitter", type=float, default=None)
    parser_strategy.add_argument("--external-poll-jitter", type=float, default=None)
    parser_strategy.add_argument("--telemetry-db", default=None)
    parser_strategy.add_argument("--no-telemetry", action="store_true", default=None)
    parser_strategy.add_argument("--source-grading", action="store_true", default=None)
    parser_strategy.add_argument("--no-source-grading", action="store_false", dest="source_grading")
    parser_strategy.add_argument("--source-grading-interval", type=int, default=None)
    parser_strategy.add_argument("--source-grading-window-hours", type=int, default=None)
    parser_strategy.add_argument("--source-grade-max-age-hours", type=float, default=None)
    parser_strategy.set_defaults(func=command_strategy)

    parser_shell = subparsers.add_parser("shell", help="launch the fallback-friendly slash-command shell")
    parser_shell.set_defaults(func=command_shell)

    parser_objectives = subparsers.add_parser("objectives", help="list registered training objectives")
    parser_objectives.set_defaults(func=command_objectives)

    parser_model_blueprint = subparsers.add_parser(
        "model-blueprint",
        help="show the research-backed model and training roadmap",
    )
    parser_model_blueprint.add_argument(
        "--risk-level",
        choices=["conservative", "regular", "aggressive", "default", "balanced", "risky"],
        default=None,
        help="filter the roadmap to one risk level",
    )
    parser_model_blueprint.add_argument(
        "--implemented-only",
        action="store_true",
        help="hide research-only, blocked, and sandbox model families",
    )
    parser_model_blueprint.add_argument("--json", action="store_true")
    parser_model_blueprint.set_defaults(func=command_model_blueprint)

    parser_train_suite = subparsers.add_parser(
        "train-suite", help="train one advanced model per objective (Conservative/Regular/Aggressive)",
    )
    parser_train_suite.add_argument("--input", default="data/historical_market.json")
    parser_train_suite.add_argument("--output-dir", default="data")
    parser_train_suite.add_argument(
        "--symbol",
        default=None,
        help="explicit asset identity for durable terminal governance; omission is research-only",
    )
    parser_train_suite.add_argument("--starting-cash", type=float, default=1000.0)
    parser_train_suite.add_argument(
        "--objective", action="append", default=None,
        help="restrict suite to named objective(s); repeat to list multiple.",
    )
    parser_train_suite.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="parallel candidate workers; defaults to available CPU cores",
    )
    parser_train_suite.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="training backend override; GPU backends run candidates sequentially to protect VRAM",
    )
    parser_train_suite.add_argument("--batch-size", type=int, default=8192, help="mini-batch size for GPU training")
    parser_train_suite.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="smoke/research cap per objective; default evaluates the full grid",
    )
    parser_train_suite.set_defaults(func=command_train_suite)

    parser_model_lab = subparsers.add_parser(
        "model-lab",
        help="rank liquid symbols and run profitability-gated model optimization across risk objectives",
    )
    parser_model_lab.add_argument("--output-dir", default="data/model_lab")
    parser_model_lab.add_argument("--starting-cash", type=float, default=1000.0)
    parser_model_lab.add_argument("--objective", action="append", default=None, help="objective/risk level to run; repeatable")
    parser_model_lab.add_argument("--max-symbols", type=int, default=6)
    parser_model_lab.add_argument("--max-scan", type=int, default=250)
    parser_model_lab.add_argument("--limit", type=int, default=1000, help="candles per selected symbol")
    parser_model_lab.add_argument("--quote-asset", default=None, help="override runtime quote asset for this lab run")
    parser_model_lab.add_argument("--interval", default=None, help="override runtime interval for this lab run")
    parser_model_lab.add_argument(
        "--full-history",
        action="store_true",
        help="page klines backward for each selected symbol until no older closed candles are returned",
    )
    parser_model_lab.add_argument(
        "--market-db",
        default=None,
        help="SQLite market-data database to train from instead of exchange API klines",
    )
    parser_model_lab.add_argument(
        "--require-db-data",
        action="store_true",
        help="force model-lab to train from SQLite market data; defaults to data/market_data.sqlite when --market-db is omitted",
    )
    parser_model_lab.add_argument("--market", choices=["spot", "futures"], default=None, help="override runtime market type for this lab run")
    parser_model_lab.add_argument("--compute-backend", choices=_COMPUTE_BACKEND_CHOICES, default=None)
    parser_model_lab.add_argument("--batch-size", type=int, default=8192)
    parser_model_lab.add_argument("--score-batch-size", type=int, default=None)
    parser_model_lab.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="smoke/research cap per objective; default evaluates the full grid",
    )
    parser_model_lab.add_argument(
        "--learning-feedback",
        default=None,
        help=(
            "optional learning_feedback.json artifact; default uses "
            "data/autonomous/learning_feedback.json when present"
        ),
    )
    parser_model_lab.set_defaults(func=command_model_lab)

    parser_backtest_panel = subparsers.add_parser(
        "backtest-panel", help="run a user-parameterized backtest and save a tagged report",
    )
    parser_backtest_panel.add_argument("--interval", required=True)
    parser_backtest_panel.add_argument("--market", default=None, help="override runtime market type")
    parser_backtest_panel.add_argument("--from-date", default=None)
    parser_backtest_panel.add_argument("--to-date", default=None)
    parser_backtest_panel.add_argument("--input", default="data/historical_market.json")
    parser_backtest_panel.add_argument("--model", default=None)
    parser_backtest_panel.add_argument("--objective", default=None)
    parser_backtest_panel.add_argument("--tag", default="")
    parser_backtest_panel.add_argument("--notes", default="")
    parser_backtest_panel.add_argument("--starting-cash", type=float, default=1000.0)
    parser_backtest_panel.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="feature/scoring backend override; default uses saved runtime compute_backend",
    )
    parser_backtest_panel.add_argument(
        "--execution-db",
        default=None,
        help="optional SQLite market-data DB for symbol-specific top-of-book fill stress",
    )
    parser_backtest_panel.set_defaults(func=command_backtest_panel)

    parser_autonomous = subparsers.add_parser(
        "autonomous", help="control the autonomous non-mainnet loop (start/pause/resume/stop/status)",
    )
    parser_autonomous.add_argument(
        "action", choices=["start", "pause", "resume", "stop", "status"],
        help="autonomous action to perform",
    )
    parser_autonomous.add_argument("--objective", default="conservative")
    parser_autonomous.add_argument("--model", default="data/model.json", help="model artifact used for autonomous decisions")
    parser_autonomous.add_argument("--poll-seconds", type=float, default=30.0, help="seconds between autonomous iterations")
    parser_autonomous.add_argument("--iterations", type=int, default=None, help="stop after N iterations; default runs until stopped")
    parser_autonomous.add_argument("--heartbeat-every", type=int, default=1, help="write heartbeat every N iterations")
    parser_autonomous.add_argument("--starting-cash", type=float, default=1000.0, help="reference cash for local autonomous risk stats")
    parser_autonomous.add_argument("--paper", action="store_true", default=False, help="force autonomous paper mode")
    parser_autonomous.add_argument("--live", action="store_true", default=False, help="force authenticated non-mainnet autonomous mode")
    parser_autonomous.set_defaults(func=command_autonomous)

    parser_positions = subparsers.add_parser(
        "positions", help="list autonomous open positions and P&L stats",
    )
    parser_positions.add_argument("--stats", action="store_true", help="also print realized + unrealized stats")
    parser_positions.add_argument("--learning", action="store_true", help="also print bounded post-trade learning feedback")
    parser_positions.set_defaults(func=command_positions)

    parser_close = subparsers.add_parser(
        "close",
        help="refuse unsafe ledger-only closure and direct users to autonomous stop",
    )
    parser_close.add_argument("position_id", help="position id or 'all'")
    parser_close.set_defaults(func=command_close)

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _build_client(runtime):
    return BinanceClient(
        api_key=runtime.api_key,
        api_secret=runtime.api_secret,
        testnet=runtime.testnet,
        demo=getattr(runtime, "demo", False),
        market_type=runtime.market_type,
        max_calls_per_minute=runtime.max_rate_calls_per_minute,
        recv_window_ms=getattr(runtime, "recv_window_ms", 5000),
    )


def _runtime_environment(runtime) -> str:
    if getattr(runtime, "demo", False):
        return "demo"
    return "testnet" if runtime.testnet else "mainnet"


def _allows_signed_execution(runtime) -> bool:
    return bool(runtime.testnet or getattr(runtime, "demo", False))


def _has_api_credentials(runtime) -> bool:
    api_key = str(getattr(runtime, "api_key", "") or "").strip()
    api_secret = str(getattr(runtime, "api_secret", "") or "").strip()
    return bool(api_key and api_secret)


def _model_taker_fee_compatibility_error(
    model: TrainedModel | None,
    required_taker_fee_bps: float,
) -> str | None:
    if model is None:
        return None
    raw_fee = getattr(model, "strategy_overrides", {}).get("taker_fee_bps")
    try:
        model_fee = float(raw_fee)
    except (TypeError, ValueError, OverflowError):
        return "Model promotion evidence does not record its taker-fee assumption."
    if not math.isfinite(model_fee) or model_fee < 0.0:
        return "Model promotion evidence contains an invalid taker-fee assumption."
    required = max(0.0, float(required_taker_fee_bps))
    if model_fee + 1e-12 < required:
        return (
            f"Model was evaluated at {model_fee:.4f} bps taker fees, below the required "
            f"{required:.4f} bps. Retrain and promote under current execution costs."
        )
    return None


def _credential_required_message(action: str) -> str:
    return f"{action} requires Binance API key and secret in Connection settings."


def _credential_failure_message(action: str, exc: Exception) -> str:
    return f"{action} requires valid Binance API credentials: {exc}"


def _credential_fingerprint(runtime) -> str:
    if not _has_api_credentials(runtime):
        return "missing"
    key = str(getattr(runtime, "api_key", "") or "")
    secret = str(getattr(runtime, "api_secret", "") or "")
    return hashlib.sha256(f"{key}\0{secret}".encode("utf-8")).hexdigest()


def _ensure_runtime_symbol(runtime, client) -> None:
    ensure_symbol = getattr(client, "ensure_symbol", None)
    if callable(ensure_symbol):
        ensure_symbol(runtime.symbol)
    else:
        client.ensure_btcusdc()


def _validate_runtime_connection(runtime, client) -> None:
    client.ping()
    _ensure_runtime_symbol(runtime, client)
    if _has_api_credentials(runtime):
        client.get_account()


_API_BUDGET_LIVE_START_MAX_USED_RATIO = 0.80


def _latest_api_budget_snapshot(db_path: str | Path, market_type: str) -> dict[str, object] | None:
    try:
        with MarketDataStore(db_path) as store:
            return store.latest_api_rate_limit_snapshot("binance", market_type)
    except OSError:
        return None


def _safe_int_value(value: object) -> int | None:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _api_budget_snapshot_age_ms(payload: Mapping[str, object], *, now_ms: int | None = None) -> int | None:
    generated_at = _safe_int_value(payload.get("generated_at_ms"))
    if generated_at is None:
        return None
    current = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return max(0, current - generated_at)


def _api_budget_snapshot_is_fresh(
    payload: Mapping[str, object],
    *,
    now_ms: int | None = None,
    max_age_ms: int = 90_000,
) -> bool:
    age = _api_budget_snapshot_age_ms(payload, now_ms=now_ms)
    return age is not None and 0 <= age <= max(1, int(max_age_ms))


def _store_api_budget_snapshot(
    db_path: str | Path,
    market_type: str,
    payload: Mapping[str, object],
) -> None:
    try:
        with MarketDataStore(db_path) as store:
            store.insert_api_rate_limit_snapshot(
                "binance",
                market_type,
                payload,
                ts_ms=int(payload.get("generated_at_ms") or time.time() * 1000),
            )
    except OSError:
        return


def _refresh_api_budget_report(runtime: RuntimeConfig, client, *, db_path: str | Path = "data/market_data.sqlite"):
    fetch_exchange_info = getattr(client, "get_exchange_info", None)
    exchange_info = fetch_exchange_info() if callable(fetch_exchange_info) else None
    request_info = dict(getattr(client, "last_request_info", {}) or {})
    report = build_api_budget_report(
        market_type=runtime.market_type,
        exchange_info=exchange_info if isinstance(exchange_info, Mapping) else None,
        request_info=request_info,
    )
    _store_api_budget_snapshot(db_path, runtime.market_type, report.asdict())
    return report


def _ensure_api_budget_startup_safe(
    runtime: RuntimeConfig,
    client,
    *,
    db_path: str | Path = "data/market_data.sqlite",
    max_used_ratio: float = _API_BUDGET_LIVE_START_MAX_USED_RATIO,
):
    cached = _latest_api_budget_snapshot(db_path, runtime.market_type)
    if cached is not None and _api_budget_snapshot_is_fresh(cached, max_age_ms=90_000):
        reason = api_budget_startup_block_reason(cached, max_used_ratio=max_used_ratio)
        if reason is not None:
            raise BinanceAPIError(reason)
    report = _refresh_api_budget_report(runtime, client, db_path=db_path)
    reason = api_budget_startup_block_reason(report, max_used_ratio=max_used_ratio)
    if reason is not None:
        raise BinanceAPIError(reason)
    return report


def _parse_form_bool(raw: str, default: bool) -> bool:
    token = raw.strip().lower()
    if token in {"y", "yes", "true", "1", "on"}:
        return True
    if token in {"n", "no", "false", "0", "off"}:
        return False
    return default


def _parse_optional_form_bool(raw: str) -> bool | None:
    token = raw.strip().lower()
    if not token:
        return None
    if token in {"y", "yes", "true", "1", "on"}:
        return True
    if token in {"n", "no", "false", "0", "off"}:
        return False
    raise ValueError(f"Expected yes/no/blank, got {raw!r}.")


def _parse_training_preset(raw: str) -> str:
    preset = (raw.strip().lower() or "custom")
    if preset not in _TRAINING_PRESETS:
        choices = "/".join(sorted(_TRAINING_PRESETS))
        raise ValueError(f"Preset must be one of: {choices}.")
    return preset


def _parse_strategy_profile(raw: str) -> str:
    profile = (raw.strip().lower() or "custom")
    if profile not in _STRATEGY_PROFILES:
        choices = "/".join(sorted(_STRATEGY_PROFILES))
        raise ValueError(f"Profile must be one of: {choices}.")
    return profile


def _strategy_with_profile(strategy: StrategyConfig, profile: str) -> StrategyConfig:
    """Return a strategy with one canonical risk profile applied."""

    canonical = _parse_strategy_profile(profile)
    return StrategyConfig(**{**strategy.asdict(), **_STRATEGY_PROFILES[canonical]})


def _apply_training_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = _parse_training_preset(str(getattr(args, "preset", "custom") or "custom"))
    for key, value in _TRAINING_PRESETS[preset].items():
        setattr(args, key, value)
    setattr(args, "preset", preset)
    return args


def _parse_form_int(raw: str, *, label: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    value = default if not raw.strip() else int(raw.strip())
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return value


def _parse_optional_form_int(raw: str, *, label: str, minimum: int | None = None, maximum: int | None = None) -> int | None:
    if not raw.strip():
        return None
    return _parse_form_int(raw, label=label, default=0, minimum=minimum, maximum=maximum)


def _parse_form_float(
    raw: str,
    *,
    label: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = default if not raw.strip() else float(raw.strip())
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return value


def _unchanged_form_value(payload: dict[str, str], key: str, current: object) -> bool:
    return payload.get(key, "").strip() == str(current)


def _profile_field_value(
    profile: str,
    payload: dict[str, str],
    key: str,
    current: _T,
    parser: Callable[[str], _T],
) -> _T | None:
    if profile != "custom" and _unchanged_form_value(payload, key, current):
        return None
    return parser(payload[key])


async def _ui_edit_runtime(ui, current: RuntimeConfig) -> RuntimeConfig:
    from .tui import FormField

    payload = await ui.form(
        "Connection settings",
        [
            FormField("market_type", "Market type [spot/futures]", current.market_type),
            FormField("interval", "Kline interval", current.interval),
            FormField("testnet", "Use Binance testnet [yes/no]", "yes" if current.testnet else "no"),
            FormField(
                "demo",
                "Use Binance Demo Trading API [yes/no]",
                "yes" if getattr(current, "demo", False) else "no",
            ),
            FormField("api_key", "Binance API key [blank keeps current]", "", password=True),
            FormField("api_secret", "Binance API secret [blank keeps current]", "", password=True),
            FormField("dry_run", "Paper trading mode [yes/no]", "yes" if current.dry_run else "no"),
            FormField("validate_account", "Validate credentials at startup [yes/no]", "yes" if current.validate_account else "no"),
            FormField("max_rate_calls_per_minute", "Max REST calls per minute", str(current.max_rate_calls_per_minute)),
            FormField("recv_window_ms", "Request recvWindow (ms, 1-60000)", str(getattr(current, "recv_window_ms", 5000))),
        ],
    )
    if payload is None:
        return current
    market_type = payload["market_type"].strip().lower()
    if market_type not in {"spot", "futures"}:
        market_type = current.market_type
    interval = payload["interval"].strip() or current.interval
    testnet = _parse_form_bool(payload["testnet"], current.testnet)
    demo = _parse_form_bool(payload.get("demo", ""), getattr(current, "demo", False))
    api_key = payload["api_key"].strip() or current.api_key
    api_secret = payload["api_secret"].strip() or current.api_secret
    dry_run = _parse_form_bool(payload["dry_run"], current.dry_run)
    validate_account = _parse_form_bool(payload["validate_account"], current.validate_account)
    max_rate = _parse_form_int(
        payload["max_rate_calls_per_minute"],
        label="Max REST calls per minute",
        default=current.max_rate_calls_per_minute,
        minimum=1,
    )
    recv_window_ms = _parse_form_int(
        payload.get("recv_window_ms", "5000"),
        label="Request recvWindow",
        default=getattr(current, "recv_window_ms", 5000),
        minimum=1,
        maximum=60000,
    )
    return RuntimeConfig(
        symbol="BTCUSDC",
        interval=interval,
        market_type=market_type,
        testnet=testnet,
        demo=demo,
        api_key=api_key,
        api_secret=api_secret,
        dry_run=dry_run,
        validate_account=validate_account,
        max_rate_calls_per_minute=max_rate,
        recv_window_ms=recv_window_ms,
        compute_backend=getattr(current, "compute_backend", "cpu"),
        managed_usdc=getattr(current, "managed_usdc", 0.0),
        managed_btc=getattr(current, "managed_btc", 0.0),
    )


async def _ui_edit_strategy_args(ui, cfg: StrategyConfig) -> argparse.Namespace:
    from .tui import FormField

    selected_features = await ui.multi_select(
        "Model feature selection",
        list(FEATURE_NAMES),
        list(cfg.enabled_features),
        help_text="Space toggles a feature. Save commits the selection used during retraining.",
    )
    if selected_features is None:
        return argparse.Namespace(
            profile="custom",
            leverage=None,
            risk=None,
            max_position=None,
            stop=None,
            take=None,
            cooldown=None,
            max_open=None,
            max_trades_per_day=None,
            signal_threshold=None,
            max_drawdown=None,
            max_daily_loss=None,
            max_session_loss=None,
            max_consecutive_losses=None,
            max_network_errors=None,
            recovery_cooldown_seconds=None,
            taker_fee_bps=None,
            slippage_bps=None,
            label_threshold=None,
            model_lookback=None,
            training_epochs=None,
            confidence_beta=None,
            feature_window_short=None,
            feature_window_long=None,
            set_features=None,
            enable_feature=None,
            disable_feature=None,
            external_signals=None,
            external_signal_max_adjustment=None,
            external_signal_min_providers=None,
            external_signal_ttl=None,
            external_signal_timeout=None,
            external_news_ai=None,
            external_news_ai_model=None,
            external_news_ai_url=None,
            external_news_ai_timeout=None,
            external_news_provider_limit=None,
            external_provider_parallelism=None,
            external_provider_jitter=None,
            external_poll_jitter=None,
            telemetry_db=None,
            no_telemetry=None,
            source_grading=None,
            source_grading_interval=None,
            source_grading_window_hours=None,
            source_grade_max_age_hours=None,
        )
    enabled_features = normalize_enabled_features(selected_features)
    payload = await ui.form(
        "Strategy settings",
        [
            FormField("profile", "Risk profile [custom/conservative/balanced/active]", "custom"),
            FormField("leverage", "Leverage", str(cfg.leverage)),
            FormField("risk", "Risk per trade", str(cfg.risk_per_trade)),
            FormField("max_position", "Max position percent", str(cfg.max_position_pct)),
            FormField("stop", "Stop loss percent", str(cfg.stop_loss_pct)),
            FormField("take", "Take profit percent", str(cfg.take_profit_pct)),
            FormField("cooldown", "Cooldown minutes", str(cfg.cooldown_minutes)),
            FormField("max_open", "Max open positions", str(cfg.max_open_positions)),
            FormField("max_trades_per_day", "Max trades per day", str(cfg.max_trades_per_day)),
            FormField("signal_threshold", "Signal threshold", str(cfg.signal_threshold)),
            FormField("max_drawdown", "Max drawdown limit", str(cfg.max_drawdown_limit)),
            FormField("max_daily_loss", "Max daily loss", str(cfg.max_daily_loss_pct)),
            FormField("max_session_loss", "Max session loss", str(cfg.max_session_loss_pct)),
            FormField("max_consecutive_losses", "Max consecutive losses", str(cfg.max_consecutive_losses)),
            FormField("max_network_errors", "Network errors before halt", str(cfg.max_network_errors)),
            FormField("recovery_cooldown_seconds", "Reconnect recovery seconds", str(cfg.recovery_cooldown_seconds)),
            FormField("taker_fee_bps", "Taker fee bps", str(cfg.taker_fee_bps)),
            FormField("slippage_bps", "Slippage bps", str(cfg.slippage_bps)),
            FormField("label_threshold", "Label threshold", str(cfg.label_threshold)),
            FormField("model_lookback", "Model lookback rows", str(cfg.model_lookback)),
            FormField("training_epochs", "Training epochs", str(cfg.training_epochs)),
            FormField("confidence_beta", "Confidence beta", str(cfg.confidence_beta)),
            FormField("feature_window_short", "Feature window short", str(cfg.feature_windows[0])),
            FormField("feature_window_long", "Feature window long", str(cfg.feature_windows[1])),
            FormField("external_signals", "External signals [yes/no]", str(cfg.external_signals_enabled)),
            FormField("external_signal_max_adjustment", "External max score adjustment", str(cfg.external_signal_max_adjustment)),
            FormField("external_signal_min_providers", "External min providers", str(cfg.external_signal_min_providers)),
            FormField("external_signal_ttl", "External cache TTL seconds", str(cfg.external_signal_ttl_seconds)),
            FormField("external_signal_timeout", "External timeout seconds", str(cfg.external_signal_timeout_seconds)),
            FormField("external_news_ai", "Ollama news AI [yes/no]", str(cfg.external_news_ai_enabled)),
            FormField("external_news_ai_model", "Ollama news model", str(cfg.external_news_ai_model)),
            FormField("external_news_provider_limit", "News provider limit", str(cfg.external_signal_news_provider_limit)),
            FormField("external_provider_parallelism", "News provider parallelism", str(cfg.external_signal_provider_parallelism)),
            FormField("external_provider_jitter", "Provider jitter seconds", str(cfg.external_signal_provider_jitter_seconds)),
            FormField("external_poll_jitter", "Poll jitter seconds", str(cfg.external_signal_poll_jitter_seconds)),
            FormField("telemetry_db", "Raw telemetry DB", str(cfg.telemetry_db_path)),
            FormField("source_grading", "Hourly source grading [yes/no]", str(cfg.source_grading_enabled)),
            FormField("source_grading_interval", "Source grading interval seconds", str(cfg.source_grading_interval_seconds)),
            FormField("source_grade_max_age_hours", "Source grade max age hours", str(cfg.source_grade_max_age_hours)),
        ],
    )
    if payload is None:
        return argparse.Namespace(
            profile="custom",
            leverage=None,
            risk=None,
            max_position=None,
            stop=None,
            take=None,
            cooldown=None,
            max_open=None,
            max_trades_per_day=None,
            signal_threshold=None,
            max_drawdown=None,
            max_daily_loss=None,
            max_session_loss=None,
            max_consecutive_losses=None,
            max_network_errors=None,
            recovery_cooldown_seconds=None,
            taker_fee_bps=None,
            slippage_bps=None,
            label_threshold=None,
            model_lookback=None,
            training_epochs=None,
            confidence_beta=None,
            feature_window_short=None,
            feature_window_long=None,
            set_features=None,
            enable_feature=None,
            disable_feature=None,
            external_signals=None,
            external_signal_max_adjustment=None,
            external_signal_min_providers=None,
            external_signal_ttl=None,
            external_signal_timeout=None,
            external_news_ai=None,
            external_news_ai_model=None,
            external_news_ai_url=None,
            external_news_ai_timeout=None,
            external_news_provider_limit=None,
            external_provider_parallelism=None,
            external_provider_jitter=None,
            external_poll_jitter=None,
            telemetry_db=None,
            no_telemetry=None,
            source_grading=None,
            source_grading_interval=None,
            source_grading_window_hours=None,
            source_grade_max_age_hours=None,
        )
    profile = _parse_strategy_profile(payload["profile"])

    def field_float(key: str, current: float, label: str, *, minimum=None, maximum=None):
        return _profile_field_value(
            profile,
            payload,
            key,
            current,
            lambda raw: _parse_form_float(raw, label=label, default=current, minimum=minimum, maximum=maximum),
        )

    def field_int(key: str, current: int, label: str, *, minimum=None, maximum=None):
        return _profile_field_value(
            profile,
            payload,
            key,
            current,
            lambda raw: _parse_form_int(raw, label=label, default=current, minimum=minimum, maximum=maximum),
        )

    def field_bool(key: str, current: bool):
        return _profile_field_value(
            profile,
            payload,
            key,
            current,
            lambda raw: _parse_form_bool(raw, current),
        )

    feature_window_short = field_int("feature_window_short", cfg.feature_windows[0], "Feature window short", minimum=1)
    feature_window_floor = int(feature_window_short or cfg.feature_windows[0])
    feature_window_long = _profile_field_value(
        profile,
        payload,
        "feature_window_long",
        cfg.feature_windows[1],
        lambda raw: _parse_form_int(
            raw,
            label="Feature window long",
            default=max(cfg.feature_windows[1], feature_window_floor + 1),
            minimum=feature_window_floor + 1,
        ),
    )
    return argparse.Namespace(
        profile=profile,
        leverage=field_float("leverage", cfg.leverage, "Leverage", minimum=1.0),
        risk=field_float("risk", cfg.risk_per_trade, "Risk per trade", minimum=0.0001),
        max_position=field_float("max_position", cfg.max_position_pct, "Max position percent", minimum=0.0, maximum=1.0),
        stop=field_float("stop", cfg.stop_loss_pct, "Stop loss percent", minimum=0.0, maximum=0.99),
        take=field_float("take", cfg.take_profit_pct, "Take profit percent", minimum=0.0, maximum=0.99),
        cooldown=field_int("cooldown", cfg.cooldown_minutes, "Cooldown minutes", minimum=0),
        max_open=field_int("max_open", cfg.max_open_positions, "Max open positions", minimum=0),
        max_trades_per_day=field_int("max_trades_per_day", cfg.max_trades_per_day, "Max trades per day", minimum=0),
        signal_threshold=field_float("signal_threshold", cfg.signal_threshold, "Signal threshold", minimum=0.01, maximum=0.99),
        max_drawdown=field_float("max_drawdown", cfg.max_drawdown_limit, "Max drawdown limit", minimum=0.0, maximum=1.0),
        max_daily_loss=field_float("max_daily_loss", cfg.max_daily_loss_pct, "Max daily loss", minimum=0.0, maximum=0.25),
        max_session_loss=field_float("max_session_loss", cfg.max_session_loss_pct, "Max session loss", minimum=0.0, maximum=0.50),
        max_consecutive_losses=field_int(
            "max_consecutive_losses",
            cfg.max_consecutive_losses,
            "Max consecutive losses",
            minimum=0,
        ),
        max_network_errors=field_int("max_network_errors", cfg.max_network_errors, "Network errors before halt", minimum=1),
        recovery_cooldown_seconds=field_int(
            "recovery_cooldown_seconds",
            cfg.recovery_cooldown_seconds,
            "Reconnect recovery seconds",
            minimum=0,
        ),
        taker_fee_bps=field_float("taker_fee_bps", cfg.taker_fee_bps, "Taker fee bps", minimum=0.0),
        slippage_bps=field_float("slippage_bps", cfg.slippage_bps, "Slippage bps", minimum=0.0),
        label_threshold=field_float("label_threshold", cfg.label_threshold, "Label threshold", minimum=0.0001),
        model_lookback=field_int("model_lookback", cfg.model_lookback, "Model lookback rows", minimum=10),
        training_epochs=field_int("training_epochs", cfg.training_epochs, "Training epochs", minimum=1),
        confidence_beta=field_float("confidence_beta", cfg.confidence_beta, "Confidence beta", minimum=0.0, maximum=1.0),
        feature_window_short=feature_window_short,
        feature_window_long=feature_window_long,
        set_features=",".join(enabled_features),
        enable_feature=None,
        disable_feature=None,
        external_signals=field_bool("external_signals", cfg.external_signals_enabled),
        external_signal_max_adjustment=field_float(
            "external_signal_max_adjustment",
            cfg.external_signal_max_adjustment,
            "External max score adjustment",
            minimum=0.0,
            maximum=0.20,
        ),
        external_signal_min_providers=field_int(
            "external_signal_min_providers",
            cfg.external_signal_min_providers,
            "External min providers",
            minimum=0,
            maximum=120,
        ),
        external_signal_ttl=field_int(
            "external_signal_ttl",
            cfg.external_signal_ttl_seconds,
            "External cache TTL seconds",
            minimum=0,
        ),
        external_signal_timeout=field_float(
            "external_signal_timeout",
            cfg.external_signal_timeout_seconds,
            "External timeout seconds",
            minimum=0.1,
            maximum=30.0,
        ),
        external_news_ai=field_bool("external_news_ai", cfg.external_news_ai_enabled),
        external_news_ai_model=payload["external_news_ai_model"],
        external_news_ai_url=None,
        external_news_ai_timeout=None,
        external_news_provider_limit=field_int(
            "external_news_provider_limit",
            cfg.external_signal_news_provider_limit,
            "News provider limit",
            minimum=0,
            maximum=120,
        ),
        external_provider_parallelism=field_int(
            "external_provider_parallelism",
            cfg.external_signal_provider_parallelism,
            "News provider parallelism",
            minimum=1,
            maximum=64,
        ),
        external_provider_jitter=field_float(
            "external_provider_jitter",
            cfg.external_signal_provider_jitter_seconds,
            "Provider jitter seconds",
            minimum=0.0,
            maximum=30.0,
        ),
        external_poll_jitter=field_float(
            "external_poll_jitter",
            cfg.external_signal_poll_jitter_seconds,
            "Poll jitter seconds",
            minimum=0.0,
            maximum=60.0,
        ),
        telemetry_db=payload["telemetry_db"],
        no_telemetry=None,
        source_grading=field_bool("source_grading", cfg.source_grading_enabled),
        source_grading_interval=field_int(
            "source_grading_interval",
            cfg.source_grading_interval_seconds,
            "Source grading interval seconds",
            minimum=60,
        ),
        source_grading_window_hours=None,
        source_grade_max_age_hours=field_float(
            "source_grade_max_age_hours",
            cfg.source_grade_max_age_hours,
            "Source grade max age hours",
            minimum=0.0,
            maximum=8760.0,
        ),
    )


def _recent_artifacts(*, base_dir: Path = Path("data"), limit: int = 8) -> list[Path]:
    if not base_dir.exists():
        return []
    paths = [path for path in base_dir.glob("*.json") if path.is_file()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[:limit]


def _artifact_summary(path: Path) -> str:
    return load_artifact_preview(path)


def _show_recent_artifacts() -> int:
    artifacts = _recent_artifacts()
    if not artifacts:
        print("No recent artifacts under data/.")
        return 0
    print("Recent artifacts:")
    for path in artifacts:
        print(f"- {_artifact_summary(path)}")
    return 0


def _render_operator_report(
    *,
    with_account: bool,
    doctor: bool,
    online: bool,
    input_path: str,
    model_path: str,
    width: int = 90,
) -> str:
    sections = [render_dashboard(_dashboard_snapshot(with_account=with_account), width=width)]
    if doctor:
        ok, lines = _readiness_report(input_path=input_path, model_path=model_path, online=online)
        status = "ok" if ok else "fix"
        sections.append("\n".join([f"Readiness report ({status})", *lines]))
    return "\n\n".join(sections)


def _account_overview_lines(runtime) -> list[str]:  # skipcq: PY-R1000
    if not _has_api_credentials(runtime):
        return [_credential_required_message("Account balances")]
    try:
        client = _build_client(runtime)
        account = client.get_account()
    except BinanceAPIError as exc:
        return [f"Account balances failed: {exc}"]
    balances_payload = account.get("balances", []) if isinstance(account, dict) else []
    assets_payload = account.get("assets", []) if isinstance(account, dict) else []
    positions_payload = account.get("positions", []) if isinstance(account, dict) else []
    balances = balances_payload if isinstance(balances_payload, list) else []
    assets = assets_payload if isinstance(assets_payload, list) else []
    positions = positions_payload if isinstance(positions_payload, list) else []
    quote_asset, base_asset = _fund_asset_labels(runtime)
    important_assets = {quote_asset, base_asset, "USDC", "USDT"}
    interesting = []
    for item in balances:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", ""))
        free = str(item.get("free", "0"))
        locked = str(item.get("locked", "0"))
        if asset in important_assets or free not in {"0", "0.0", "0.00000000"} or locked not in {"0", "0.0", "0.00000000"}:
            interesting.append(f"{asset}: free={free} locked={locked}")
    for item in assets:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", ""))
        wallet = str(item.get("walletBalance", item.get("availableBalance", "0")))
        available = str(item.get("availableBalance", "0"))
        unrealized = str(item.get("unrealizedProfit", "0"))
        if asset in important_assets or wallet not in {"0", "0.0", "0.00000000"} or unrealized not in {"0", "0.0", "0.00000000"}:
            interesting.append(f"{asset}: wallet={wallet} available={available} unrealized={unrealized}")
    for item in positions:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", ""))
        amount = str(item.get("positionAmt", item.get("positionAmount", "0")))
        entry = str(item.get("entryPrice", "0"))
        unrealized = str(item.get("unrealizedProfit", "0"))
        if amount not in {"0", "0.0", "0.00000000"} or unrealized not in {"0", "0.0", "0.00000000"}:
            interesting.append(f"{symbol}: position={amount} entry={entry} unrealized={unrealized}")
    if not interesting:
        return [
            f"market={runtime.market_type} environment={_runtime_environment(runtime)} testnet={runtime.testnet}",
            "No non-zero balances found.",
        ]
    return [
        f"market={runtime.market_type} environment={_runtime_environment(runtime)} testnet={runtime.testnet}",
        *interesting[:20],
    ]


def _show_account_overview() -> int:
    runtime = load_runtime()
    if not _has_api_credentials(runtime):
        print(_credential_required_message("Account balances"), file=sys.stderr)
        return 2
    lines = _account_overview_lines(runtime)
    print("Account balances")
    for line in lines:
        print(f"- {line}" if ":" in line and not line.startswith("market=") else line)
    if lines and lines[0].startswith("Account balances failed:"):
        return 2
    return 0


def _dashboard_snapshot(*, with_account: bool) -> DashboardSnapshot:
    runtime = load_runtime()
    strategy = load_strategy()
    notes = [
        "Operate the system from the interactive console actions and modal forms.",
        "Use authenticated execution only on testnet and only after checking runtime state.",
    ]
    return DashboardSnapshot(
        runtime=runtime.public_dict(),
        strategy=strategy.asdict(),
        artifacts=[_artifact_summary(path) for path in _recent_artifacts()],
        account_lines=_account_overview_lines(runtime) if with_account else ["Run Account balances after Connect to fetch balances."],
        notes=notes,
    )


def _connection_status_line() -> str:
    runtime = load_runtime()
    environment = _runtime_environment(runtime)
    market = f"{runtime.market_type}/{environment}"
    mode = "paper-default" if runtime.dry_run else f"{environment}-live-default"
    checked_at = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    try:
        client = _build_client(runtime)
        client.ping()
        server_time = client.get_exchange_time()
    except BinanceAPIError as exc:
        return f"Connection {checked_at}: public endpoint unreachable {market}; {exc}"
    server_label = "server-time ok" if server_time is not None else "server-time response ok"
    if not _has_api_credentials(runtime):
        return f"Connection {checked_at}: public endpoint reachable {market} {mode}; credentials missing"
    if not _allows_signed_execution(runtime):
        return (
            f"Connection {checked_at}: public endpoint reachable {market} {mode}; "
            "credentials saved, signed validation locked until testnet or demo is enabled"
        )
    if not bool(getattr(runtime, "validate_account", True)):
        return f"Connection {checked_at}: public endpoint reachable {market} {mode}; credentials saved, not validated"
    try:
        account = client.get_account()
    except BinanceAPIError as exc:
        return f"Connection {checked_at}: authentication failed {market} {mode}; {exc}"
    auth_label = "authenticated" if isinstance(account, dict) else "auth response ok"
    return f"Connection {checked_at}: {auth_label} {market} {mode}; {server_label}"


def _readiness_report(*, input_path: str, model_path: str, online: bool = False) -> tuple[bool, list[str]]:  # skipcq: PY-R1000
    runtime = load_runtime()
    strategy = load_strategy()
    checks: list[tuple[bool, str, str]] = []

    def add(ok: bool, label: str, detail: str) -> None:
        checks.append((ok, label, detail))

    environment = _runtime_environment(runtime)
    safe_execution = _allows_signed_execution(runtime)
    mode = "paper" if runtime.dry_run else f"authenticated {environment}"
    add(
        safe_execution,
        "safety target",
        f"{environment} enabled" if safe_execution else "testnet/demo is disabled",
    )
    add(runtime.market_type in {"spot", "futures"}, "market type", runtime.market_type)
    add(
        runtime.dry_run or bool(runtime.api_key and runtime.api_secret),
        "execution credentials",
        f"{mode}; credentials {'present' if runtime.api_key and runtime.api_secret else 'not configured'}",
    )
    add(0.01 <= strategy.signal_threshold <= 0.99, "signal threshold", f"{strategy.signal_threshold:.3f}")
    add(strategy.risk_per_trade > 0 and strategy.max_position_pct > 0, "risk sizing", f"risk={strategy.risk_per_trade} max_position={strategy.max_position_pct}")
    add(bool(strategy.enabled_features), "feature set", ",".join(strategy.enabled_features))

    data_file = Path(input_path)
    candles = None
    if data_file.exists():
        candles = _load_rows_for_command(str(data_file), label="Readiness data load failed")
        candle_count = len(candles) if candles is not None else 0
        add(candle_count > max(strategy.feature_windows), "training data", f"{candle_count} candles at {data_file}")
    else:
        add(False, "training data", f"missing {data_file}")

    model_file = Path(model_path)
    if model_file.exists():
        try:
            model, model_kind = _load_readiness_model(model_file, strategy)
        except (OSError, ValueError) as exc:
            add(False, "model artifact", f"{model_file} is not usable with current strategy ({exc})")
        else:
            add(True, "model artifact", f"{model_file} dim={model.feature_dim} kind={model_kind}")
            effective_strategy = apply_model_strategy_overrides(strategy, model)
            if effective_strategy.asdict() != strategy.asdict():
                add(
                    True,
                    "model strategy overlay",
                    (
                        f"threshold={effective_strategy.signal_threshold:.3f} "
                        f"risk={effective_strategy.risk_per_trade:.4f} "
                        f"stop={effective_strategy.stop_loss_pct:.4f} "
                        f"take={effective_strategy.take_profit_pct:.4f}"
                    ),
                )
            quality_score = getattr(model, "quality_score", None)
            if quality_score is not None:
                warnings = list(getattr(model, "quality_warnings", []) or [])
                warning_text = "; ".join(str(item) for item in warnings[:2]) or "none"
                add(
                    float(quality_score) >= 0.45,
                    "model quality",
                    f"score={float(quality_score):.2f} warnings={warning_text}",
                )
            probability_brier = getattr(model, "probability_brier_after", None)
            if probability_brier is not None:
                probability_temperature = float(getattr(model, "probability_temperature", 1.0) or 1.0)
                probability_ece = getattr(model, "probability_ece_after", None)
                probability_detail = (
                    f"temperature={probability_temperature:.2f} "
                    f"brier={float(probability_brier):.3f}"
                )
                if probability_ece is not None:
                    probability_detail += f" ece={float(probability_ece):.3f}"
                add(
                    float(probability_brier) <= 0.35
                    and (probability_ece is None or float(probability_ece) <= 0.20),
                    "probability calibration",
                    probability_detail,
                )
            if candles is not None:
                rows = _readiness_model_rows(candles, effective_strategy, model)
                if rows:
                    drift = feature_drift_report(rows[-min(50, len(rows)):], model)
                    drift_warning = "; ".join(drift.warnings[:2]) or "none"
                    add(
                        drift.status != "fail",
                        "feature drift",
                        (
                            f"status={drift.status} max_z={drift.max_abs_z:.2f} "
                            f"outliers={drift.outlier_fraction:.1%} warnings={drift_warning}"
                        ),
                    )
    else:
        add(False, "model artifact", f"missing {model_file}")

    if online:
        line = _connection_status_line()
        authenticated = "authenticated" in line or "auth response ok" in line
        add(authenticated and "failed" not in line and "unreachable" not in line, "exchange connectivity", line)

    lines = [f"[{'ok' if ok else 'fix'}] {label}: {detail}" for ok, label, detail in checks]
    return all(ok for ok, _label, _detail in checks), lines


_COMPUTE_BACKEND_CHOICES = ("cpu", "cuda", "rocm", "directml", "mps", "auto")


def _runtime_quote_asset(runtime: RuntimeConfig) -> str:
    configured = str(getattr(runtime, "quote_asset", "") or "").strip().upper()
    symbol = str(getattr(runtime, "symbol", "") or "").strip().upper()
    if configured in SUPPORTED_MAJOR_QUOTE_ASSETS and symbol.endswith(configured):
        return configured
    for quote in sorted(SUPPORTED_MAJOR_QUOTE_ASSETS, key=len, reverse=True):
        if symbol.endswith(quote):
            return quote
    return "USDC"


def _fund_asset_labels(runtime: RuntimeConfig) -> tuple[str, str]:
    quote_asset = _runtime_quote_asset(runtime)
    base_asset = symbol_base_for_supported_quote(getattr(runtime, "symbol", ""), quote_asset=quote_asset)
    return quote_asset, base_asset or "BTC"


def _workflow_compute_backend(
    runtime: RuntimeConfig,
    requested: object,
    *,
    workflow: str,
) -> tuple[str, BackendInfo]:
    backend_name = str(requested or runtime.compute_backend or default_compute_backend()).strip().lower()
    if backend_name not in _COMPUTE_BACKEND_CHOICES:
        raise ValueError(f"unknown compute backend {backend_name!r}")
    info = resolve_backend(backend_name)
    if info.kind == "cpu":
        detail = f" ({info.reason})" if info.reason else ""
        print(
            f"warning: {workflow} is running in CPU-only mode{detail}; "
            "training, retraining, and backtest scoring will be much slower and AI features are disabled.",
            file=sys.stderr,
        )
    return backend_name, info


def _account_free_balances(account: object, tracked_assets: Iterable[str] | None = None) -> dict[str, float]:
    asset_names = tuple(dict.fromkeys(str(asset or "").strip().upper() for asset in (tracked_assets or ("USDC", "BTC"))))
    balances = {asset: 0.0 for asset in asset_names if asset}
    if not isinstance(account, dict):
        return balances
    spot_balances = account.get("balances", [])
    if isinstance(spot_balances, list):
        for item in spot_balances:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("asset", "")).upper()
            if asset in balances:
                balances[asset] += max(0.0, _safe_float(item.get("free")))
    futures_assets = account.get("assets", [])
    if isinstance(futures_assets, list):
        for item in futures_assets:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("asset", "")).upper()
            if asset in balances:
                value = item.get("availableBalance", item.get("walletBalance", 0.0))
                balances[asset] = max(balances[asset], max(0.0, _safe_float(value)))
    return balances


def _load_exchange_funds(runtime) -> dict[str, float]:
    if not _has_api_credentials(runtime):
        raise BinanceAPIError(_credential_required_message("Funds"))
    account = _build_client(runtime).get_account()
    return _account_free_balances(account, _fund_asset_labels(runtime))


def _funds_summary(runtime, balances: Mapping[str, float] | None = None) -> str:
    quote_asset, base_asset = _fund_asset_labels(runtime)
    caps = f"Trading caps: {quote_asset}={runtime.managed_usdc:.4f} {base_asset}={runtime.managed_btc:.8f}"
    if not _has_api_credentials(runtime):
        return f"API credentials missing. Exchange-backed Funds disabled. {caps}"
    if balances is None:
        return f"Exchange balance not loaded. {caps}"
    quote_free = max(0.0, _safe_float(balances.get(quote_asset, 0.0)))
    base_free = max(0.0, _safe_float(balances.get(base_asset, 0.0)))
    return f"Exchange free: {quote_asset}={quote_free:.4f} {base_asset}={base_free:.8f}. {caps}"


def _apply_funds_change(
    action: str,
    amount: float,
    *,
    balances: Mapping[str, float] | None = None,
) -> tuple[object, str]:
    """Apply an exchange-backed Funds-menu allocation cap mutation."""
    runtime = load_runtime()
    quote_asset, base_asset = _fund_asset_labels(runtime)
    if action == "clear":
        runtime.managed_usdc = 0.0
        runtime.managed_btc = 0.0
        save_runtime(runtime)
        return runtime, "Cleared trading caps. No exchange balances were changed."
    if action in {"deposit_usdc", "withdraw_usdc", "deposit_btc", "withdraw_btc", "reset"}:
        return runtime, "Funds no longer supports local deposit, withdraw, or reset. Configure credentials and set exchange-backed caps."
    if action not in {"sync", "set_usdc", "set_btc"}:
        return runtime, f"Unknown funds action {action!r}."
    if balances is None:
        return runtime, _credential_required_message("Funds")
    if action == "sync":
        runtime.managed_usdc = max(0.0, _safe_float(balances.get(quote_asset, 0.0)))
        runtime.managed_btc = max(0.0, _safe_float(balances.get(base_asset, 0.0)))
        msg = "Synced trading caps from exchange free balances."
    else:
        asset = quote_asset if action == "set_usdc" else base_asset
        available = max(0.0, _safe_float(balances.get(asset, 0.0)))
        requested = max(0.0, float(amount))
        cap = min(requested, available)
        if action == "set_usdc":
            runtime.managed_usdc = cap
            msg = f"Set {quote_asset} trading cap to {cap:.4f}."
        else:
            runtime.managed_btc = cap
            msg = f"Set {base_asset} trading cap to {cap:.8f}."
        if requested > available:
            msg += f" Requested amount was capped to exchange free balance {available:.8f} {asset}."
    save_runtime(runtime)
    return runtime, msg


async def _ui_funds_menu(ui) -> int:
    from .tui import FormField

    while True:
        runtime = load_runtime()
        has_credentials = _has_api_credentials(runtime)
        quote_asset, base_asset = _fund_asset_labels(runtime)
        options = (
            [
                ("sync", "Use exchange free balances as caps"),
                ("set_usdc", f"Set {quote_asset} trading cap"),
                ("set_btc", f"Set {base_asset} trading cap"),
                ("clear", "Clear trading caps"),
                ("show", "Refresh exchange-backed allocation"),
                ("close", "Close"),
            ]
            if has_credentials
            else [
                ("show", "Show credential requirement"),
                ("close", "Close"),
            ]
        )
        choice = await ui.menu(
            "Trading caps - exchange-backed asset limits",
            options,
            help_text=(
                f"{_funds_summary(runtime)}. Trading caps reads Binance {quote_asset}/{base_asset} balances "
                "and stores maximum strategy allocation caps; it never deposits, withdraws, or simulates money."
            ),
        )
        if choice in (None, "close"):
            return 0
        if not has_credentials:
            ui.append_log(_credential_required_message("Funds"))
            continue
        if choice == "clear":
            _, msg = _apply_funds_change("clear", 0.0)
            ui.append_log(msg)
            continue
        try:
            balances = await ui.run_blocking(_load_exchange_funds, runtime)
        except BinanceAPIError as exc:
            ui.append_log(_credential_failure_message("Funds", exc))
            continue
        if choice == "show":
            ui.append_log(_funds_summary(runtime, balances))
            continue
        if choice == "sync":
            _, msg = _apply_funds_change("sync", 0.0, balances=balances)
            ui.append_log(msg)
            continue
        is_base_asset = choice == "set_btc"
        unit = base_asset if is_base_asset else quote_asset
        current_cap = runtime.managed_btc if is_base_asset else runtime.managed_usdc
        payload = await ui.form(
            f"Set {unit} trading cap",
            [
                FormField(
                    "amount",
                    f"Maximum {unit} the strategy may use",
                    f"{current_cap:.8f}" if is_base_asset else f"{current_cap:.4f}",
                ),
            ],
        )
        if payload is None:
            continue
        try:
            amount = _parse_form_float(
                payload["amount"],
                label=f"Amount ({unit})",
                default=0.0,
                minimum=0.0,
            )
        except ValueError as exc:
            ui.append_log(f"Funds change rejected: {exc}")
            continue
        _, msg = _apply_funds_change(choice, amount, balances=balances)
        ui.append_log(msg)


async def _ui_edit_execution(ui) -> int:
    from .tui import FormField

    cfg = load_strategy()
    payload = await ui.form(
        "Order settings",
        [
            FormField(
                "order_type",
                "Order type [MARKET]",
                str(getattr(cfg, "order_type", "MARKET")),
            ),
            FormField(
                "time_in_force",
                "Time in force [GTC; market orders ignore this]",
                str(getattr(cfg, "time_in_force", "GTC")),
            ),
            FormField(
                "post_only",
                "Post-only [no; unsupported for market orders]",
                "yes" if getattr(cfg, "post_only", False) else "no",
            ),
            FormField(
                "reduce_only_on_close",
                "Reduce-only when closing [yes/no]",
                "yes" if getattr(cfg, "reduce_only_on_close", True) else "no",
            ),
        ],
    )
    if payload is None:
        ui.append_log("Order settings cancelled.")
        return 0
    order_type = payload["order_type"].strip().upper() or "MARKET"
    if order_type != "MARKET":
        ui.append_log(f"Unsupported order type {order_type!r}; using MARKET.")
        order_type = "MARKET"
    tif = payload["time_in_force"].strip().upper() or "GTC"
    if tif not in {"GTC", "IOC", "FOK"}:
        ui.append_log(f"Unsupported timeInForce {tif!r}; keeping {cfg.time_in_force!r}.")
        tif = cfg.time_in_force
    post_only = _parse_form_bool(payload["post_only"], getattr(cfg, "post_only", False))
    if post_only:
        ui.append_log("Post-only is not compatible with live market execution; using no.")
        post_only = False
    reduce_only = _parse_form_bool(
        payload["reduce_only_on_close"],
        getattr(cfg, "reduce_only_on_close", True),
    )
    cfg.order_type = order_type
    cfg.time_in_force = tif
    cfg.post_only = post_only
    cfg.reduce_only_on_close = reduce_only
    save_strategy(cfg)
    ui.append_log(
        f"Saved execution: order_type={order_type} tif={tif} "
        f"post_only={post_only} reduce_only_on_close={reduce_only}"
    )
    return 0


async def _ui_edit_compute(ui) -> int:
    from .compute import describe_backend, resolve_backend
    from .tui import FormField

    runtime = load_runtime()
    current = getattr(runtime, "compute_backend", "cpu")
    payload = await ui.form(
        "Compute backend",
        [
            FormField(
                "backend",
                "Backend [cpu / cuda / rocm / directml / mps / auto]",
                current,
            ),
        ],
    )
    if payload is None:
        ui.append_log("Compute backend selection cancelled.")
        return 0
    requested = payload["backend"].strip().lower() or "cpu"
    if requested not in _COMPUTE_BACKEND_CHOICES:
        ui.append_log(f"Unknown backend {requested!r}; keeping {current!r}.")
        return 2
    info = await ui.run_blocking(resolve_backend, requested)
    runtime.compute_backend = requested
    if info.kind == "cpu" and runtime.ai_enabled:
        runtime.ai_enabled = False
        ui.append_log("AI features disabled because the selected compute backend is CPU-only.")
    save_runtime(runtime)
    ui.append_log(
        f"Saved compute_backend={requested}. Runtime status: {describe_backend(info)}"
    )
    if info.kind == "cpu":
        ui.append_log("CPU-only mode remains usable, but training/backtesting will be slower and AI cannot run.")
    return 0


async def _ui_settings_menu(ui, mark_credentials: Callable[[str], None] | None = None) -> int:
    while True:
        choice = await ui.menu(
            "All settings",
            [
                ("runtime", "Connection - API, market, safety mode"),
                ("strategy", "Strategy - risk, signals, model behavior"),
                ("execution", "Orders - type and close behavior"),
                ("compute", "Compute - CPU / GPU / auto"),
                ("close", "Close"),
            ],
            help_text="Choose what to configure. Up/Down to choose, Enter to open, Escape to close.",
        )
        if choice in (None, "close"):
            return 0
        if choice == "runtime":
            current = load_runtime()
            try:
                next_runtime = await _ui_edit_runtime(ui, current)
            except ValueError as exc:
                ui.append_log(f"Connection settings invalid: {exc}")
                continue
            if next_runtime == current:
                ui.append_log("Connection settings cancelled.")
                continue
            save_runtime(next_runtime)
            if mark_credentials is not None:
                mark_credentials("unchecked" if _has_api_credentials(next_runtime) else "missing")
            ui.append_log("Connection settings saved.")
            if next_runtime.validate_account and _has_api_credentials(next_runtime):
                try:
                    client = _build_client(next_runtime)
                    await ui.run_blocking(_validate_runtime_connection, next_runtime, client)
                except BinanceAPIError as exc:
                    if mark_credentials is not None:
                        mark_credentials("invalid")
                    ui.append_log(f"Connection validation failed: {exc}")
                    continue
                if mark_credentials is not None:
                    mark_credentials("valid")
                ui.append_log("Connection credentials validated.")
            continue
        if choice == "strategy":
            try:
                args = await _ui_edit_strategy_args(ui, load_strategy())
            except ValueError as exc:
                ui.append_log(f"Strategy settings invalid: {exc}")
                continue
            if args.set_features is None and args.leverage is None and getattr(args, "profile", "custom") == "custom":
                ui.append_log("Strategy update cancelled.")
                continue
            await ui.run_blocking(command_strategy, args)
            continue
        if choice == "execution":
            await _ui_edit_execution(ui)
            continue
        if choice == "compute":
            await _ui_edit_compute(ui)
            continue


def _tui_actions(credential_state: dict[str, str] | None = None):  # skipcq: PY-R1000
    from .tui import FormField, TUIAction

    def _credential_status() -> str:
        runtime = load_runtime()
        if not _has_api_credentials(runtime):
            return "missing"
        if credential_state is None:
            return "valid"
        fingerprint = _credential_fingerprint(runtime)
        if credential_state.get("fingerprint") != fingerprint:
            return "unchecked"
        return credential_state.get("status", "unchecked")

    def _mark_credentials(status: str) -> None:
        if credential_state is None:
            return
        runtime = load_runtime()
        credential_state["fingerprint"] = _credential_fingerprint(runtime)
        credential_state["status"] = "missing" if not _has_api_credentials(runtime) else status

    def _credential_lock_reason(action: str) -> str:
        runtime = load_runtime()
        if not _allows_signed_execution(runtime):
            return f"{action} is locked. Signed actions require testnet=true or demo=true."
        status = _credential_status()
        if status == "missing":
            return f"{action} is locked. Add Binance API key and secret in Connection settings first."
        if status == "invalid":
            return f"{action} is locked. The saved Binance credentials failed validation; replace them in Connection settings."
        if status == "unchecked":
            return f"{action} is locked. Run Connect after saving credentials, then try again."
        return f"{action} is locked until Binance credentials validate."

    def _connect_enabled() -> bool:
        status = _credential_status()
        return status in {"unchecked", "valid", "invalid", "unavailable"}

    def _signed_action_enabled() -> bool:
        return _credential_status() == "valid" and _allows_signed_execution(load_runtime())

    def _make_disabled_reason(action_title: str) -> Callable[[], str]:
        def disabled_reason() -> str:
            return _credential_lock_reason(action_title)

        return disabled_reason

    def _make_action(
        key: str,
        title: str,
        description: str,
        run,
        *,
        aliases: tuple[str, ...] = (),
        credentials: bool = False,
    ):
        if not credentials:
            return TUIAction(key, title, description, run, aliases=aliases)
        enabled = _connect_enabled if title == "Connect" else _signed_action_enabled
        return TUIAction(
            key,
            title,
            description,
            run,
            enabled=enabled,
            disabled_reason=_make_disabled_reason(title),
            aliases=aliases,
        )

    def _credentials_ready(ui, action: str) -> bool:
        if not _has_api_credentials(load_runtime()):
            ui.append_log(_credential_required_message(action))
            return False
        if action != "Connect" and not _allows_signed_execution(load_runtime()):
            ui.append_log(_credential_lock_reason(action))
            return False
        if credential_state is not None and action != "Connect" and _credential_status() != "valid":
            ui.append_log(_credential_lock_reason(action))
            return False
        return True

    async def _overview(ui):
        include_account = credential_state is None or _credential_status() == "valid"
        ui.append_log(await ui.run_blocking(lambda: render_dashboard(_dashboard_snapshot(with_account=include_account))))
        return 0

    async def _help(ui):
        ui.append_log(
            "\n".join(
                [
                    "Operator help - simple-ai-trading",
                    "==================================",
                    "",
                    "Scope: BTC/ETH/SOL spot/futures trading on Binance testnet or Demo Trading only.",
                    "",
                    "First-time setup",
                    "----------------",
                    "  1. Connection settings: paste your Binance testnet API key and secret.",
                    "  2. Connect: validate those credentials.",
                    f"  3. Trading caps: choose how much {base_asset} / {quote_asset} the strategy may use.",
                    "  4. Safety check: confirm safety flags, data, model, and connectivity.",
                    "",
                    "End-to-end paper run",
                    "--------------------",
                    "  1. Build full setup: download data, train, evaluate, and backtest.",
                    "  2. Paper trading: run the strategy without placing real orders.",
                    "  3. Full report: print dashboard, artifacts, and safety summary.",
                    "",
                    "Manual pipeline (full control)",
                    "------------------------------",
                    "  Download market data -> Strategy settings -> Train AI model -> Evaluate model -> Backtest strategy -> Optimize strategy",
                    "",
                    "Authenticated testnet execution",
                    "-------------------------------",
                    "  * Always run Safety check first.",
                    "  * Test order is the smallest signed BUY/SELL check.",
                    "  * Testnet trading runs the strategy with signed orders against testnet/demo.",
                    "  * Trading caps can never exceed exchange free balances.",
                    "",
                    "Keyboard",
                    "--------",
                    "  Up / Down           always move the command or modal menu selection",
                    "  Enter               run the selected command",
                    "  r                   refresh the dashboard snapshot",
                    "  <  >                shrink / grow the command list",
                    "  -  +                shrink / grow the activity log",
                    "  Ctrl-L              clear the activity log",
                    "  q                   quit",
                    "  Inside forms: Tab cycles fields, Enter advances or saves, Escape cancels.",
                    "",
                    "Centralized configuration",
                    "-------------------------",
                    "  All settings opens: Connection, Strategy, Orders, Compute backend.",
                    "  Trading caps is exchange-backed: it reads balances and sets caps, never deposits or withdraws.",
                    "  Compute backend selects CPU (default), CUDA, ROCm, DirectML, MPS, or auto-detect.",
                    "",
                    "Safety",
                    "------",
                    "  testnet=true is the default; demo=true selects Binance Demo Trading endpoints.",
                    "  Paper mode never places real orders, even when credentials are present.",
                    "  Credentials are stored at ~/.config/simple_ai_trading/runtime.json (mode 600).",
                ]
            )
        )
        return 0

    async def _runtime(ui):
        current = load_runtime()
        try:
            next_runtime = await _ui_edit_runtime(ui, current)
        except ValueError as exc:
            print(f"Connection settings invalid: {exc}", file=sys.stderr)
            return 2
        if next_runtime == current:
            print("Connection settings cancelled.")
            return 0
        save_runtime(next_runtime)
        _mark_credentials("unchecked")
        if next_runtime.validate_account and _has_api_credentials(next_runtime):
            client = _build_client(next_runtime)
            try:
                await ui.run_blocking(_validate_runtime_connection, next_runtime, client)
            except BinanceAPIError as exc:
                _mark_credentials("invalid")
                print(f"Configuration saved, but validation failed: {exc}", file=sys.stderr)
                return 2
            _mark_credentials("valid")
        print("Runtime config saved to", config_paths()["runtime"])
        print(
            f"market={next_runtime.market_type} environment={_runtime_environment(next_runtime)} "
            f"testnet={next_runtime.testnet} demo={getattr(next_runtime, 'demo', False)} paper={next_runtime.dry_run}"
        )
        return 0

    async def _strategy(ui):
        try:
            args = await _ui_edit_strategy_args(ui, load_strategy())
        except ValueError as exc:
            print(f"Strategy settings invalid: {exc}", file=sys.stderr)
            return 2
        if args.set_features is None and args.leverage is None and getattr(args, "profile", "custom") == "custom":
            print("Strategy update cancelled.")
            return 0
        return await ui.run_blocking(command_strategy, args)

    async def _connect(_ui):
        if not _credentials_ready(_ui, "Connect"):
            return 2
        result = await _ui.run_blocking(command_connect, argparse.Namespace())
        _mark_credentials("valid" if result == 0 else "invalid")
        return result

    async def _doctor(ui):
        payload = await ui.form(
            "Readiness check",
            [
                FormField("input", "Training input path", "data/historical_market.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("online", "Include exchange connectivity [yes/no]", "yes"),
            ],
        )
        if payload is None:
            print("Readiness check cancelled.")
            return 0
        return await ui.run_blocking(
            command_doctor,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_market.json",
                model=payload["model"].strip() or "data/model.json",
                online=_parse_form_bool(payload["online"], True),
            ),
        )

    async def _account(ui):
        if not _credentials_ready(ui, "Account balances"):
            return 2
        return await ui.run_blocking(_show_account_overview)

    async def _audit(ui):
        payload = await ui.form(
            "Data/model audit",
            [
                FormField("input", "Training input path", "data/historical_market.json"),
                FormField("model", "Model path", "data/model.json"),
            ],
        )
        if payload is None:
            print("Data/model audit cancelled.")
            return 0
        return await ui.run_blocking(
            command_audit,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_market.json",
                model=payload["model"].strip() or "data/model.json",
            ),
        )

    async def _fetch(ui):
        runtime = load_runtime()
        max_batch_size = 1500 if runtime.market_type == "futures" else 1000
        payload = await ui.form(
            "Download market data",
            [
                FormField("limit", "Fetch limit", "500"),
                FormField("batch_size", f"Klines per request [max {max_batch_size}]", "1000"),
                FormField("output", "Candle output path", "data/historical_market.json"),
            ],
        )
        if payload is None:
            print("Market data download cancelled.")
            return 0
        try:
            limit = _parse_form_int(payload["limit"], label="Fetch limit", default=500, minimum=1)
            batch_size = _parse_form_int(payload.get("batch_size", "1000"), label="Klines per request", default=1000, minimum=1, maximum=max_batch_size)
        except ValueError as exc:
            print(f"Fetch settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_fetch,
            argparse.Namespace(
                symbol=runtime.symbol,
                interval=runtime.interval,
                limit=limit,
                batch_size=batch_size,
                output=payload["output"].strip() or "data/historical_market.json",
            ),
        )

    async def _train(ui):
        payload = await ui.form(
            "Train AI model",
            [
                FormField("input", "Training input path", "data/historical_market.json"),
                FormField("output", "Model output path", "data/model.json"),
                FormField("preset", "Preset [custom/quick/balanced/thorough]", "custom"),
                FormField("epochs", "Training epochs", "250"),
                FormField("learning_rate", "Learning rate", "0.05"),
                FormField("l2_penalty", "L2 penalty", "0.0001"),
                FormField("seed", "Training seed", "7"),
                FormField("walk_forward", "Run walk-forward validation [yes/no]", "no"),
                FormField("walk_forward_train", "Walk-forward train window", "300"),
                FormField("walk_forward_test", "Walk-forward test window", "60"),
                FormField("walk_forward_step", "Walk-forward step", "30"),
                FormField("calibrate_threshold", "Calibrate threshold [yes/no]", "yes"),
            ],
        )
        if payload is None:
            print("AI model training cancelled.")
            return 0
        try:
            preset = _parse_training_preset(payload["preset"])
            epochs = _parse_form_int(payload["epochs"], label="Training epochs", default=250, minimum=1)
            learning_rate = _parse_form_float(payload.get("learning_rate", "0.05"), label="Learning rate", default=0.05, minimum=0.000001)
            l2_penalty = _parse_form_float(payload.get("l2_penalty", "0.0001"), label="L2 penalty", default=0.0001, minimum=0.0)
            seed = _parse_form_int(payload["seed"], label="Training seed", default=7, minimum=0)
            walk_forward_train = _parse_form_int(payload["walk_forward_train"], label="Walk-forward train window", default=300, minimum=2)
            walk_forward_test = _parse_form_int(payload["walk_forward_test"], label="Walk-forward test window", default=60, minimum=1)
            walk_forward_step = _parse_form_int(payload["walk_forward_step"], label="Walk-forward step", default=30, minimum=1)
        except ValueError as exc:
            print(f"Training settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_train,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_market.json",
                output=payload["output"].strip() or "data/model.json",
                preset=preset,
                epochs=epochs,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                seed=seed,
                walk_forward=_parse_form_bool(payload["walk_forward"], False),
                walk_forward_train=walk_forward_train,
                walk_forward_test=walk_forward_test,
                walk_forward_step=walk_forward_step,
                calibrate_threshold=_parse_form_bool(payload["calibrate_threshold"], True),
            ),
        )

    async def _tune(ui):
        payload = await ui.form(
            "Optimize strategy",
            [
                FormField("input", "Tune input path", "data/historical_market.json"),
                FormField("window_mode", "Window mode [all/lookback/range]", "all"),
                FormField("lookback_days", "Lookback days", "30"),
                FormField("from_date", "From date YYYY-MM-DD", ""),
                FormField("to_date", "To date YYYY-MM-DD", ""),
                FormField("save_best", "Persist the best strategy [yes/no]", "no"),
                FormField("min_risk", "Minimum risk", "0.002"),
                FormField("max_risk", "Maximum risk", "0.02"),
                FormField("steps", "Grid steps", "5"),
                FormField("min_leverage", "Minimum leverage", "1.0"),
                FormField("max_leverage", "Maximum leverage", "20.0"),
                FormField("min_threshold", "Minimum threshold", "0.52"),
                FormField("max_threshold", "Maximum threshold", "0.88"),
                FormField("min_take", "Minimum take profit", "0.01"),
                FormField("max_take", "Maximum take profit", "0.06"),
                FormField("min_stop", "Minimum stop loss", "0.008"),
                FormField("max_stop", "Maximum stop loss", "0.04"),
            ],
        )
        if payload is None:
            print("Strategy optimization cancelled.")
            return 0
        mode = payload["window_mode"].strip().lower()
        lookback_days = None
        from_date = None
        to_date = None
        try:
            if mode == "lookback":
                lookback_days = _parse_form_int(payload["lookback_days"], label="Lookback days", default=30, minimum=1)
            elif mode == "range":
                from_date = payload["from_date"].strip() or None
                to_date = payload["to_date"].strip() or None
            elif mode not in {"", "all"}:
                raise ValueError("Window mode must be all, lookback, or range.")
            steps = _parse_form_int(payload["steps"], label="Grid steps", default=5, minimum=1)
            min_risk = _parse_form_float(payload["min_risk"], label="Minimum risk", default=0.002, minimum=0.0001)
            max_risk = _parse_form_float(payload["max_risk"], label="Maximum risk", default=0.02, minimum=0.0001)
            min_leverage = _parse_form_float(payload["min_leverage"], label="Minimum leverage", default=1.0, minimum=1.0)
            max_leverage = _parse_form_float(payload["max_leverage"], label="Maximum leverage", default=20.0, minimum=1.0)
            min_threshold = _parse_form_float(payload["min_threshold"], label="Minimum threshold", default=0.52, minimum=0.01, maximum=0.99)
            max_threshold = _parse_form_float(payload["max_threshold"], label="Maximum threshold", default=0.88, minimum=0.01, maximum=0.99)
            min_take = _parse_form_float(payload["min_take"], label="Minimum take profit", default=0.01, minimum=0.0, maximum=0.99)
            max_take = _parse_form_float(payload["max_take"], label="Maximum take profit", default=0.06, minimum=0.0, maximum=0.99)
            min_stop = _parse_form_float(payload["min_stop"], label="Minimum stop loss", default=0.008, minimum=0.0, maximum=0.99)
            max_stop = _parse_form_float(payload["max_stop"], label="Maximum stop loss", default=0.04, minimum=0.0, maximum=0.99)
        except ValueError as exc:
            print(f"Optimization settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_tune,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_market.json",
                save_best=_parse_form_bool(payload["save_best"], False),
                min_risk=min_risk,
                max_risk=max_risk,
                steps=steps,
                min_leverage=min_leverage,
                max_leverage=max_leverage,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                min_take=min_take,
                max_take=max_take,
                min_stop=min_stop,
                max_stop=max_stop,
                lookback_days=lookback_days,
                from_date=from_date,
                to_date=to_date,
            ),
        )

    async def _backtest(ui):
        payload = await ui.form(
            "Backtest strategy",
            [
                FormField("input", "Backtest input path", "data/historical_market.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("start_cash", "Starting cash", "1000"),
            ],
        )
        if payload is None:
            print("Strategy backtest cancelled.")
            return 0
        try:
            start_cash = _parse_form_float(payload["start_cash"], label="Starting cash", default=1000.0, minimum=1.0)
        except ValueError as exc:
            print(f"Backtest settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_backtest,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_market.json",
                model=payload["model"].strip() or "data/model.json",
                start_cash=start_cash,
            ),
        )

    async def _evaluate(ui):
        payload = await ui.form(
            "Evaluate model",
            [
                FormField("input", "Evaluation input path", "data/historical_market.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("threshold", "Evaluation threshold [blank=strategy default]", ""),
                FormField("calibrate_threshold", "Calibrate threshold [yes/no]", "no"),
            ],
        )
        if payload is None:
            print("Evaluation cancelled.")
            return 0
        threshold_raw = payload["threshold"].strip()
        try:
            threshold = float(threshold_raw) if threshold_raw else None
        except ValueError as exc:
            print(f"Evaluation settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_evaluate,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_market.json",
                model=payload["model"].strip() or "data/model.json",
                threshold=threshold,
                calibrate_threshold=_parse_form_bool(payload["calibrate_threshold"], False),
            ),
        )

    async def _prepare(ui):
        runtime = load_runtime()
        max_batch_size = 1500 if runtime.market_type == "futures" else 1000
        payload = await ui.form(
            "Build full setup",
            [
                FormField("historical", "Historical candle path", "data/historical_market.json"),
                FormField("model", "Model artifact path", "data/model.json"),
                FormField("limit", "Fetch limit", "500"),
                FormField("batch_size", f"Klines per request [max {max_batch_size}]", "1000"),
                FormField("preset", "Training preset [custom/quick/balanced/thorough]", "balanced"),
                FormField("epochs", "Training epochs [blank=preset]", ""),
                FormField("learning_rate", "Learning rate", "0.05"),
                FormField("l2_penalty", "L2 penalty", "0.0001"),
                FormField("seed", "Training seed", "7"),
                FormField("walk_forward", "Walk-forward validation [yes/no/blank=preset]", ""),
                FormField("walk_forward_train", "Walk-forward train window [blank=preset]", ""),
                FormField("walk_forward_test", "Walk-forward test window [blank=preset]", ""),
                FormField("walk_forward_step", "Walk-forward step [blank=preset]", ""),
                FormField("calibrate_threshold", "Calibrate threshold [yes/no/blank=preset]", ""),
                FormField("start_cash", "Backtest starting cash", "1000"),
                FormField("online_doctor", "Include exchange connectivity in final check [yes/no]", "no"),
            ],
        )
        if payload is None:
            print("Build full setup cancelled.")
            return 0
        historical = payload["historical"].strip() or "data/historical_market.json"
        model = payload["model"].strip() or "data/model.json"
        try:
            limit = _parse_form_int(payload["limit"], label="Fetch limit", default=500, minimum=1)
            batch_size = _parse_form_int(payload.get("batch_size", "1000"), label="Klines per request", default=1000, minimum=1, maximum=max_batch_size)
            preset = _parse_training_preset(payload["preset"])
            epochs = _parse_optional_form_int(payload.get("epochs", ""), label="Training epochs", minimum=1)
            learning_rate = _parse_form_float(payload.get("learning_rate", "0.05"), label="Learning rate", default=0.05, minimum=0.000001)
            l2_penalty = _parse_form_float(payload.get("l2_penalty", "0.0001"), label="L2 penalty", default=0.0001, minimum=0.0)
            seed = _parse_form_int(payload["seed"], label="Training seed", default=7, minimum=0)
            walk_forward_train = _parse_optional_form_int(payload.get("walk_forward_train", ""), label="Walk-forward train window", minimum=2)
            walk_forward_test = _parse_optional_form_int(payload.get("walk_forward_test", ""), label="Walk-forward test window", minimum=1)
            walk_forward_step = _parse_optional_form_int(payload.get("walk_forward_step", ""), label="Walk-forward step", minimum=1)
            walk_forward = _parse_optional_form_bool(payload.get("walk_forward", ""))
            calibrate_threshold_opt = _parse_optional_form_bool(payload.get("calibrate_threshold", ""))
            start_cash = _parse_form_float(payload["start_cash"], label="Backtest starting cash", default=1000.0, minimum=1.0)
        except ValueError as exc:
            print(f"Setup settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_prepare,
            argparse.Namespace(
                historical=historical,
                model=model,
                limit=limit,
                batch_size=batch_size,
                preset=preset,
                epochs=epochs,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                seed=seed,
                walk_forward=walk_forward,
                walk_forward_train=walk_forward_train,
                walk_forward_test=walk_forward_test,
                walk_forward_step=walk_forward_step,
                calibrate_threshold=calibrate_threshold_opt,
                start_cash=start_cash,
                online_doctor=_parse_form_bool(payload["online_doctor"], False),
            ),
        )

    async def _paper(ui):
        payload = await ui.form(
            "Paper trading",
            [
                FormField("model", "Model path", "data/model.json"),
                FormField("steps", "Paper trading steps", "20"),
                FormField("sleep", "Sleep seconds", "5"),
                FormField("retrain_interval", "Retrain interval", "0"),
                FormField("retrain_window", "Retrain window", "300"),
                FormField("retrain_min_rows", "Retrain minimum rows", "240"),
            ],
        )
        if payload is None:
            print("Paper trading cancelled.")
            return 0
        try:
            steps = _parse_form_int(payload["steps"], label="Paper trading steps", default=20, minimum=1)
            sleep = _parse_form_int(payload["sleep"], label="Sleep seconds", default=5, minimum=0)
            retrain_interval = _parse_form_int(payload["retrain_interval"], label="Retrain interval", default=0, minimum=0)
            retrain_window = _parse_form_int(payload["retrain_window"], label="Retrain window", default=300, minimum=1)
            retrain_min_rows = _parse_form_int(payload["retrain_min_rows"], label="Retrain minimum rows", default=240, minimum=1)
        except ValueError as exc:
            print(f"Paper trading settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_live,
            argparse.Namespace(
                steps=steps,
                model=payload["model"].strip() or "data/model.json",
                sleep=sleep,
                leverage=None,
                retrain_interval=retrain_interval,
                retrain_window=retrain_window,
                retrain_min_rows=retrain_min_rows,
                paper=True,
                live=False,
            ),
        )

    async def _live(ui):
        if not _credentials_ready(ui, "Testnet trading"):
            return 2
        payload = await ui.form(
            "Testnet trading",
            [
                FormField("model", "Model path", "data/model.json"),
                FormField("steps", "Live steps", "1"),
                FormField("sleep", "Sleep seconds", "5"),
                FormField("retrain_interval", "Retrain interval", "0"),
                FormField("retrain_window", "Retrain window", "300"),
                FormField("retrain_min_rows", "Retrain minimum rows", "240"),
            ],
        )
        if payload is None:
            print("Testnet trading cancelled.")
            return 0
        try:
            steps = _parse_form_int(payload["steps"], label="Live steps", default=1, minimum=1)
            sleep = _parse_form_int(payload["sleep"], label="Sleep seconds", default=5, minimum=0)
            retrain_interval = _parse_form_int(payload["retrain_interval"], label="Retrain interval", default=0, minimum=0)
            retrain_window = _parse_form_int(payload["retrain_window"], label="Retrain window", default=300, minimum=1)
            retrain_min_rows = _parse_form_int(payload["retrain_min_rows"], label="Retrain minimum rows", default=240, minimum=1)
        except ValueError as exc:
            print(f"Testnet trading settings invalid: {exc}", file=sys.stderr)
            return 2
        model = payload["model"].strip() or "data/model.json"
        runtime = load_runtime()
        environment = _runtime_environment(runtime)
        if not await ui.confirm(
            f"Run authenticated {environment} execution with model={model}, steps={steps}, sleep={sleep}s?"
        ):
            print(f"{environment.capitalize()} execution cancelled.")
            return 0
        return await ui.run_blocking(
            command_live,
            argparse.Namespace(
                steps=steps,
                model=model,
                sleep=sleep,
                leverage=None,
                retrain_interval=retrain_interval,
                retrain_window=retrain_window,
                retrain_min_rows=retrain_min_rows,
                paper=False,
                live=True,
            ),
        )

    async def _roundtrip(ui):
        if not _credentials_ready(ui, "Test order"):
            return 2
        payload = await ui.form(
            "Test order",
            [
                FormField("quantity", "Order quantity", "0.00008"),
                FormField("mode", "Mode [auto/buy-sell/sell-buy]", "auto"),
            ],
        )
        if payload is None:
            print("Test order cancelled.")
            return 0
        try:
            quantity = _parse_form_float(payload["quantity"], label="Order quantity", default=0.00008, minimum=0.00001)
        except ValueError as exc:
            print(f"Test order settings invalid: {exc}", file=sys.stderr)
            return 2
        mode = payload["mode"].strip().lower() or "auto"
        if mode not in {"auto", "buy-sell", "sell-buy"}:
            print("Test order mode must be auto, buy-sell, or sell-buy.", file=sys.stderr)
            return 2
        runtime = load_runtime()
        if not await ui.confirm(
            f"Place {mode} spot {_runtime_environment(runtime)} roundtrip for quantity={quantity:.8f}?"
        ):
            print("Test order cancelled.")
            return 0
        return await ui.run_blocking(
            command_spot_roundtrip,
            argparse.Namespace(quantity=quantity, mode=mode, yes=True),
        )

    async def _report(ui):
        payload = await ui.form(
            "Full report",
            [
                FormField("input", "Training input path", "data/historical_market.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("readiness", "Include readiness report [yes/no]", "yes"),
                FormField("online", "Include exchange connectivity [yes/no]", "no"),
                FormField("account", "Include account state [yes/no]", "no"),
            ],
        )
        if payload is None:
            print("Full report cancelled.")
            return 0
        include_account = _parse_form_bool(payload["account"], False)
        if include_account and not _credentials_ready(ui, "Full report account section"):
            return 2
        return await ui.run_blocking(
            command_report,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_market.json",
                model=payload["model"].strip() or "data/model.json",
                doctor=_parse_form_bool(payload["readiness"], True),
                online=_parse_form_bool(payload["online"], False),
                account=include_account,
            ),
        )

    async def _funds(ui):
        if not _credentials_ready(ui, "Trading caps"):
            return 2
        return await _ui_funds_menu(ui)

    async def _settings(ui):
        return await _ui_settings_menu(ui, mark_credentials=_mark_credentials)

    quote_asset, base_asset = _fund_asset_labels(load_runtime())
    return [
        _make_action("1", "Connect", "Validate the saved Binance testnet credentials and unlock account-only actions.", _connect, credentials=True),
        _make_action("2", "Dashboard", "Show the current setup, strategy, model, and recent run artifacts in the activity log.", _overview, aliases=("Overview",)),
        _make_action("3", "Account balances", f"Read authenticated {base_asset} and {quote_asset} balances from Binance.", _account, credentials=True, aliases=("Account",)),
        _make_action("4", "Safety check", "Verify safety flags, training data, model compatibility, and optional exchange reachability.", _doctor, aliases=("Readiness check",)),
        _make_action("5", "Data/model audit", "Check candle quality, feature stability, model metadata, and risk posture without network calls.", _audit, aliases=("Local audit",)),
        _make_action("6", "Trading caps", f"Read exchange balances and set the maximum {base_asset} / {quote_asset} the strategy may use.", _funds, credentials=True, aliases=("Funds",)),
        _make_action("7", "Download market data", "Download fresh market candles into the local dataset.", _fetch, aliases=("Fetch candles",)),
        _make_action("8", "Train AI model", "Train or retrain the prediction model on cached market data.", _train, aliases=("Train model",)),
        _make_action("9", "Evaluate model", "Score the saved model on cached candles and inspect threshold quality.", _evaluate, aliases=("Evaluate",)),
        _make_action("10", "Backtest strategy", "Simulate trades on cached candles and report PnL, fees, and drawdown.", _backtest, aliases=("Backtest",)),
        _make_action("11", "Optimize strategy", "Search risk, threshold, take-profit, and stop-loss settings over a chosen history window.", _tune, aliases=("Tune strategy",)),
        _make_action("12", "Build full setup", "Download data, train, evaluate, backtest, audit, then run safety checks.", _prepare, aliases=("Prepare system",)),
        _make_action("13", "Paper trading", "Run the live loop without placing orders; useful before signed testnet trading.", _paper, aliases=("Paper loop",)),
        _make_action("14", "Testnet trading", "Run authenticated non-mainnet execution with signed testnet or demo orders.", _live, credentials=True, aliases=("Testnet loop",)),
        _make_action("15", "Test order", "Place a minimal BUY/SELL roundtrip on spot testnet or demo.", _roundtrip, credentials=True, aliases=("Spot roundtrip",)),
        _make_action("16", "Full report", "Print dashboard, recent artifacts, safety checks, and optional account state.", _report, aliases=("Operator report",)),
        # Backwards-compatible aliases for direct configuration shortcuts.
        _make_action("17", "Connection settings", "Edit API keys, market type, testnet/demo mode, paper mode, and request limits.", _runtime, aliases=("Runtime settings",)),
        _make_action("18", "Strategy settings", "Edit risk, thresholds, model windows, enabled features, and external signal behavior.", _strategy),
        _make_action("19", "All settings", "Open one settings hub for connection, strategy, order execution, and CPU/GPU backend.", _settings, aliases=("Settings",)),
        _make_action("20", "Help", "Show the plain-language workflow, keyboard shortcuts, safety notes, and setup guide.", _help),
    ]


def command_menu(_: argparse.Namespace) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("Interactive console requires a real terminal (TTY).", file=sys.stderr)
        return 2
    if not supports_ansi_terminal(sys.stdout):
        print(
            "Interactive console needs ANSI/virtual-terminal support. "
            "Use Windows Terminal, or run `simple-ai-trading shell` for the plain fallback.",
            file=sys.stderr,
        )
        return 2

    from .tui import launch_tui

    initial_runtime = load_runtime()
    credential_state = {
        "fingerprint": _credential_fingerprint(initial_runtime),
        "status": "missing" if not _has_api_credentials(initial_runtime) else "unchecked",
    }

    def menu_connection_status_line() -> str:
        line = _connection_status_line()
        runtime = load_runtime()
        credential_state["fingerprint"] = _credential_fingerprint(runtime)
        if not _has_api_credentials(runtime):
            credential_state["status"] = "missing"
        elif "authentication failed" in line:
            credential_state["status"] = "invalid"
        elif "authenticated" in line or "auth response ok" in line:
            credential_state["status"] = "valid"
        else:
            credential_state["status"] = "unavailable"
        return line

    return launch_tui(
        title="simple-ai-trading interactive console",
        actions=_tui_actions(credential_state),
        snapshot_provider=_menu_dashboard_snapshot,
        connection_provider=menu_connection_status_line,
    )


def _menu_dashboard_snapshot(width: int = 72) -> str:
    return render_dashboard(_dashboard_snapshot(with_account=False), width=width)


def _load_json_candles(path: str) -> list[Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected candle list in JSON file: {path}")
    return payload


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(float(value)):
        return low
    if value < low:
        return low
    if value > high:
        return high
    return value


def _jittered_seconds(base_seconds: float, jitter_seconds: float) -> float:
    base = max(0.0, float(base_seconds))
    jitter = max(0.0, float(jitter_seconds))
    return base + (_JITTER_RANDOM.uniform(0.0, jitter) if jitter else 0.0)


def _rows_from_json(path: str):
    candles_raw = _load_json_candles(path)

    rows: list[Candle] = []
    for item in candles_raw:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(
                Candle(
                    open_time=int(item["open_time"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item["volume"]),
                    close_time=int(item["close_time"]),
                )
            )
        except (TypeError, ValueError, KeyError):
            continue
    return rows


def _load_rows_for_command(path: str, *, label: str) -> list | None:
    try:
        return _rows_from_json(path)
    except (OSError, ValueError) as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return None


def _parse_date_boundary(raw: str, *, end_of_day: bool) -> int:
    dt = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(dt.timestamp() * 1000)


def _filter_candles_for_time_window(  # skipcq: PY-R1000
    candles: Sequence[Candle],
    *,
    lookback_days: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[Candle]:
    if lookback_days is not None and lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if lookback_days is not None and (from_date or to_date):
        raise ValueError("lookback_days cannot be combined with from_date/to_date")

    start_ms: int | None = None
    end_ms: int | None = None
    if from_date:
        start_ms = _parse_date_boundary(from_date, end_of_day=False)
    if to_date:
        end_ms = _parse_date_boundary(to_date, end_of_day=True)
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValueError("from_date must be <= to_date")

    filtered = list(candles)
    if lookback_days is not None and filtered:
        latest_close = max(int(candle.close_time) for candle in filtered)
        start_ms = latest_close - (lookback_days * 24 * 60 * 60 * 1000)

    if start_ms is not None:
        filtered = [candle for candle in filtered if int(candle.open_time) >= start_ms]
    if end_ms is not None:
        filtered = [candle for candle in filtered if int(candle.open_time) <= end_ms]
    return filtered


def _artifact_path(kind: str, *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{kind}_run_{int(time.time() * 1_000_000)}.json"


def _persist_run_artifact(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
    path = _artifact_path(kind, output_dir=output_dir)
    write_json_atomic(path, payload, indent=2, sort_keys=True)
    return path


def _public_runtime_payload(runtime) -> dict[str, object]:
    return runtime.public_dict()


def _build_model_rows(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    *,
    compute_backend: str | None = None,
) -> list[ModelRow]:
    candles = clean_candles(candles)
    return make_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        lookahead=1,
        label_threshold=strategy.label_threshold,
        enabled_features=strategy.enabled_features,
        compute_backend=compute_backend,
    )


def _effective_leverage(cfg: StrategyConfig, market_type: str) -> float:
    if market_type != "futures":
        return 1.0
    leverage = float(cfg.leverage)
    if not math.isfinite(leverage):
        return 1.0
    return float(max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, leverage)))


def _resolve_futures_leverage(runtime, cfg: StrategyConfig) -> float:
    """Resolve leverage from runtime+strategy with an exchange-side clamp when possible."""
    requested = _effective_leverage(cfg, runtime.market_type)
    if runtime.market_type != "futures":
        return requested
    if not runtime.api_key or not runtime.api_secret:
        return requested
    client = _build_client(runtime)
    try:
        max_leverage = client.get_max_leverage(runtime.symbol)
        if max_leverage > 0:
            return float(min(requested, max_leverage))
    except BinanceAPIError:
        return requested
    return requested


def _entry_leverage_for_notional(
    client,
    runtime,
    requested_leverage: float,
    notional: float,
    *,
    effective_dry_run: bool,
) -> float:
    """Return the futures entry leverage that matches the order's notional bracket."""
    if runtime.market_type != "futures":
        return 1.0
    requested = _safe_float(requested_leverage)
    if not math.isfinite(requested) or requested <= 0.0:
        requested = 1.0
    requested = max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, requested))
    if effective_dry_run:
        return requested
    clean_notional = abs(_safe_float(notional))
    if not math.isfinite(clean_notional) or clean_notional <= 0.0:
        raise BinanceAPIError("Unable to resolve futures leverage bracket without a positive entry notional")
    if not hasattr(client, "get_max_leverage_for_notional"):
        return requested
    try:
        bracket_leverage = _safe_float(client.get_max_leverage_for_notional(runtime.symbol, clean_notional))
    except BinanceAPIError:
        raise
    except Exception as exc:
        raise BinanceAPIError(f"Unable to resolve futures leverage bracket for entry notional: {exc}") from exc
    if not math.isfinite(bracket_leverage) or bracket_leverage <= 0.0:
        raise BinanceAPIError("Invalid futures leverage bracket for entry notional")
    return max(1.0, min(requested, MAX_AUTONOMOUS_LEVERAGE, bracket_leverage))


def _resolve_symbol_constraints(runtime, client) -> SymbolConstraints | None:
    try:
        return client.get_symbol_constraints(runtime.symbol)
    except BinanceAPIError:
        return None


def _strategy_feature_signature(strategy: StrategyConfig) -> str:
    return feature_signature(
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        strategy.label_threshold,
        feature_version=strategy.feature_version,
        enabled_features=strategy.enabled_features,
    )


def _advanced_objective_for_model(
    model: TrainedModel,
    strategy: StrategyConfig,
) -> tuple[str, AdvancedFeatureConfig] | None:
    model_signature = getattr(model, "feature_signature", None)
    if not model_signature:
        return None
    for objective_name in available_objectives():
        feature_cfg = default_config_for(objective_name, strategy.enabled_features)
        if advanced_feature_signature(feature_cfg) == str(model_signature):
            return objective_name, feature_cfg
    feature_cfg = advanced_config_from_signature(str(model_signature), strategy.enabled_features)
    if feature_cfg is not None:
        return "custom", feature_cfg
    return None


def _load_readiness_model(model_path: Path, strategy: StrategyConfig) -> tuple[TrainedModel, str]:
    try:
        return _load_runtime_model(model_path, strategy), "runtime"
    except ModelFeatureMismatchError as runtime_error:
        model = load_model(
            model_path,
            expected_feature_version=strategy.feature_version,
            expected_feature_dim=None,
        )
        advanced = _advanced_objective_for_model(model, strategy)
        if advanced is None:
            raise runtime_error
        objective_name, _feature_cfg = advanced
        return model, f"advanced:{objective_name}"


def _readiness_model_rows(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    model: TrainedModel,
    *,
    compute_backend: str | None = None,
) -> list[ModelRow]:
    advanced = _advanced_objective_for_model(model, strategy)
    if advanced is not None:
        _objective_name, feature_cfg = advanced
        return make_advanced_rows(candles, feature_cfg, compute_backend=compute_backend)
    return _build_model_rows(candles, strategy, compute_backend=compute_backend)


def _backtest_rows_for_model(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    model: TrainedModel,
    *,
    compute_backend: str | None = None,
) -> list[ModelRow]:
    advanced = _advanced_objective_for_model(model, strategy)
    if advanced is not None:
        _objective_name, feature_cfg = advanced
        rows = make_advanced_inference_rows(candles, feature_cfg, compute_backend=compute_backend)
        return rows if rows else _readiness_model_rows(candles, strategy, model, compute_backend=compute_backend)
    rows = make_inference_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        enabled_features=strategy.enabled_features,
        compute_backend=compute_backend,
    )
    return rows if rows else _readiness_model_rows(candles, strategy, model, compute_backend=compute_backend)


def _live_rows_for_model(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    model: TrainedModel | None,
    *,
    compute_backend: str | None = None,
) -> list[ModelRow]:
    if model is None:
        rows = make_inference_rows(
            candles,
            strategy.feature_windows[0],
            strategy.feature_windows[1],
            enabled_features=strategy.enabled_features,
            compute_backend=compute_backend,
        )
        return rows if rows else _build_model_rows(candles, strategy, compute_backend=compute_backend)
    advanced = _advanced_objective_for_model(model, strategy)
    if advanced is not None:
        _objective_name, feature_cfg = advanced
        rows = make_advanced_inference_rows(candles, feature_cfg, compute_backend=compute_backend)
        return rows if rows else _readiness_model_rows(candles, strategy, model, compute_backend=compute_backend)
    rows = make_inference_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        enabled_features=strategy.enabled_features,
        compute_backend=compute_backend,
    )
    return rows if rows else _readiness_model_rows(candles, strategy, model, compute_backend=compute_backend)


def _live_model_feature_signature(model: TrainedModel | None, strategy: StrategyConfig) -> str:
    if model is not None and _advanced_objective_for_model(model, strategy) is not None:
        return str(model.feature_signature)
    return _strategy_feature_signature(strategy)


def _load_runtime_model(model_path: Path, strategy: StrategyConfig) -> TrainedModel:
    strategy_signature = _strategy_feature_signature(strategy)
    return load_model(
        model_path,
        expected_feature_version=strategy.feature_version,
        expected_feature_signature=strategy_signature,
        expected_feature_dim=None,
    )


def _load_live_start_model(
    model_path: Path,
    strategy: StrategyConfig,
    *,
    effective_dry_run: bool,
    require_model_candidate_search: bool = False,
    require_accelerator_evidence: bool = False,
    require_live_data_evidence: bool = False,
    require_microstructure_evidence: bool = False,
    expected_symbol: str | None = None,
    expected_market_type: str | None = None,
    expected_interval: str | None = None,
    min_live_data_years: float = 1.0,
    min_live_coverage_ratio: float = 0.995,
    max_live_gap_count: int = 0,
) -> tuple[TrainedModel | None, str | None, str | None]:
    if model_path.exists():
        try:
            model = _load_readiness_model(model_path, strategy)[0]
            if isinstance(model, TrainedModel):
                assert_model_promoted(
                    model,
                    model_path=model_path,
                    require_model_candidate_search=require_model_candidate_search,
                    require_accelerator_evidence=require_accelerator_evidence,
                    require_live_data_evidence=require_live_data_evidence,
                    require_microstructure_evidence=require_microstructure_evidence,
                    expected_symbol=expected_symbol,
                    expected_market_type=expected_market_type,
                    expected_interval=expected_interval,
                    min_live_data_years=min_live_data_years,
                    min_live_coverage_ratio=min_live_coverage_ratio,
                    max_live_gap_count=max_live_gap_count,
                    require_terminal_ledger_record=not effective_dry_run,
                )
            return model, None, None
        except ModelPromotionError as exc:
            if not effective_dry_run:
                return None, f"Live mode requires a promoted model: {exc}", None
            model = _load_readiness_model(model_path, strategy)[0]
            return model, None, f"Paper mode model promotion warning: {exc}"
        except ModelLoadError as exc:
            if not effective_dry_run:
                return None, f"Live mode requires a compatible model: {exc}", None
            return None, None, f"Model load failed; regenerating: {exc}"
        except Exception:
            if not effective_dry_run:
                return None, f"Live mode requires a readable model: {model_path}", None
            return None, None, None
    if not effective_dry_run:
        return None, f"Live mode needs model file: {model_path}", None
    return None, None, None


def _target_notional(
    cash: float,
    strategy: StrategyConfig,
    market_type: str,
    *,
    leverage: float | None = None,
) -> float:
    if not math.isfinite(float(cash)) or cash <= 0:
        return 0.0
    if leverage is None:
        leverage = _effective_leverage(strategy, market_type)
    if not math.isfinite(float(leverage)):
        return 0.0
    risk_exposure = stop_loss_sized_notional_pct(strategy, market_type, leverage=leverage)
    return max(0.0, cash * risk_exposure)


def _build_order_notional(
    cash: float,
    price: float,
    cfg: StrategyConfig,
    market_type: str,
    leverage: float,
    client,
    *,
    constraints: SymbolConstraints | None = None,
) -> tuple[float, float]:
    """Build and return adjusted (notional, qty) for a desired trade.

    Returns (notional, qty) after constraints are enforced.
    """
    if not all(math.isfinite(float(value)) for value in (cash, price, leverage)):
        return 0.0, 0.0
    if price <= 0:
        return 0.0, 0.0

    requested_notional = _target_notional(cash, cfg, market_type, leverage=leverage)
    if requested_notional <= 0:
        return 0.0, 0.0

    qty = requested_notional / price
    if constraints is None:
        return requested_notional, abs(qty)

    normalized_qty, parsed_constraints = client.normalize_quantity(constraints.symbol, abs(qty))
    if normalized_qty <= 0:
        return 0.0, 0.0

    requested_notional = normalized_qty * price

    if normalized_qty < parsed_constraints.min_qty:
        return 0.0, 0.0

    if parsed_constraints.min_notional > 0 and requested_notional < parsed_constraints.min_notional:
        return 0.0, 0.0

    if parsed_constraints.max_notional > 0 and requested_notional > parsed_constraints.max_notional:
        capped_notional = parsed_constraints.max_notional
        capped_qty, _ = client.normalize_quantity(parsed_constraints.symbol, capped_notional / price)
        if capped_qty <= 0:
            return 0.0, 0.0
        requested_notional = capped_qty * price
        normalized_qty = capped_qty

    return requested_notional, abs(normalized_qty)


def _safe_day_ms(timestamp_ms: int) -> int:
    return int(timestamp_ms // (24 * 60 * 60 * 1000))


def _resolve_live_retrain_rows(
    rows: list[ModelRow],
    *,
    retrain_window: int,
    retrain_min_rows: int,
) -> list[ModelRow]:
    if len(rows) < retrain_min_rows:
        return []
    if len(rows) <= retrain_window:
        return rows
    return rows[-retrain_window:]


def _build_live_model(
    rows: list[ModelRow],
    *,
    model: TrainedModel | None = None,
    retrain_every: int,
    step: int,
    cfg: StrategyConfig,
    retrain_window: int,
    retrain_min_rows: int,
    model_feature_signature: str | None = None,
    compute_backend: str | None = None,
    batch_size: int = 8192,
) -> TrainedModel | None:
    if model is not None:
        if retrain_every <= 0:
            return model
        if retrain_every > 0 and step % retrain_every != 0:
            return model

    train_rows = _resolve_live_retrain_rows(rows, retrain_window=retrain_window, retrain_min_rows=retrain_min_rows)
    if not train_rows:
        return model

    epochs = max(20, int(cfg.training_epochs * 0.4))
    signature = model_feature_signature or _strategy_feature_signature(cfg)
    return train(
        train_rows,
        epochs=epochs,
        feature_signature=signature,
        compute_backend=compute_backend,
        batch_size=batch_size,
    )


def _directional_threshold_confidence(
    long_threshold: float | None,
    short_threshold: float | None,
    fallback: float,
) -> float:
    values: list[float] = []
    if long_threshold is not None:
        values.append(float(long_threshold))
    if short_threshold is not None:
        values.append(1.0 - float(short_threshold))
    if not values:
        values.append(float(fallback))
    return _clamp(max(values), 0.0, 1.0)


def _adjust_directional_thresholds(
    long_threshold: float | None,
    short_threshold: float | None,
    threshold_add: float,
) -> tuple[float | None, float | None]:
    add = _clamp(float(threshold_add), -1.0, 1.0)
    adjusted_long = _clamp(float(long_threshold) + add, 0.0, 1.0) if long_threshold is not None else None
    adjusted_short = _clamp(float(short_threshold) - add, 0.0, 1.0) if short_threshold is not None else None
    return adjusted_long, adjusted_short


def _score_to_direction(
    score: float,
    cfg: StrategyConfig,
    market_type: str,
    threshold: float | None = None,
    *,
    short_threshold: float | None = None,
    side_thresholds_explicit: bool = False,
) -> int:
    threshold = cfg.signal_threshold if threshold is None else _clamp(float(threshold), 0.0, 1.0)
    return market_direction_from_probability(
        score,
        threshold,
        market_type=market_type,
        short_threshold=short_threshold,
        infer_symmetric_short=not side_thresholds_explicit,
    )


def _live_entry_risk_skip(
    *,
    step: int,
    day: int,
    score: float,
    entry_risk: EntryRiskDecision,
    max_daily_trades: int,
    max_open_positions: int,
) -> tuple[str, dict[str, object]]:
    if entry_risk.code == "trade_cap":
        message = f"step {step:>2}: trade cap reached ({max_daily_trades}/day), skipping entry"
        status = "skip_trade_cap"
        event = {"step": step, "status": status, "day": day, "score": float(score)}
    elif entry_risk.code == "max_open_positions":
        message = f"step {step:>2}: max open positions reached ({max_open_positions}), skipping entry"
        status = "skip_max_open_positions"
        event = {"step": step, "status": status, "score": float(score)}
    else:
        message = f"step {step:>2}: risk gate blocked entry ({entry_risk.code})"
        status = f"skip_risk_{entry_risk.code}"
        event = {"step": step, "status": status, "score": float(score)}
    event["risk_check"] = entry_risk.asdict()
    return message, event


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _detect_existing_position(
    runtime,
    client,
    *,
    leverage: float,
    reference_price: float | None = None,
    account: object | None = None,
) -> dict[str, float | int | str] | None:  # skipcq: PY-R1000
    if account is None:
        if not hasattr(client, "get_account"):
            return None
        account = client.get_account()
    if not isinstance(account, dict):
        return None
    if runtime.market_type == "futures":
        for item in account.get("positions", []) or []:
            if not isinstance(item, dict) or item.get("symbol") != runtime.symbol:
                continue
            amount = _safe_float(item.get("positionAmt"))
            if amount == 0.0:
                continue
            entry_price = _safe_float(item.get("entryPrice")) or _safe_float(item.get("markPrice")) or reference_price or 0.0
            qty = abs(amount)
            margin = (
                _safe_float(item.get("positionInitialMargin"))
                or _safe_float(item.get("initialMargin"))
                or _safe_float(item.get("isolatedWallet"))
                or (qty * entry_price / max(1.0, leverage) if entry_price > 0.0 else 0.0)
            )
            return {
                "market": "futures",
                "side": 1 if amount > 0.0 else -1,
                "qty": qty,
                "entry_price": entry_price,
                "notional": qty * entry_price,
                "margin": margin,
            }
        return None

    _quote_asset, base_asset = _fund_asset_labels(runtime)
    managed_base = max(0.0, _safe_float(getattr(runtime, "managed_btc", 0.0)))
    if managed_base <= 0.0:
        return None
    for item in account.get("balances", []) or []:
        if not isinstance(item, dict) or item.get("asset") != base_asset:
            continue
        qty = _safe_float(item.get("free")) + _safe_float(item.get("locked"))
        if qty <= 0.0:
            continue
        qty = min(qty, managed_base)
        price = reference_price
        if price is None:
            price, _timestamp = client.get_symbol_price(runtime.symbol)
        return {
            "market": "spot",
            "side": 1,
            "qty": qty,
            "entry_price": float(price),
            "entry_price_basis": "current_mark_price_not_cost_basis",
            "notional": qty * float(price),
            "margin": qty * float(price),
        }
    return None


def _paper_or_live_order(
    client: BinanceClient,
    runtime,
    strategy: StrategyConfig,
    *,
    side: str,
    size: float,
    dry_run: bool,
    leverage: float | None = None,
    notional: float | None = None,
    reduce_only: bool = False,
    client_order_id: str | None = None,
) -> dict[str, object]:
    if leverage is None:
        leverage = _effective_leverage(strategy, runtime.market_type)
    kwargs = {"dry_run": dry_run, "leverage": leverage}
    if notional is not None and not reduce_only and hasattr(client, "get_max_leverage_for_notional"):
        kwargs["notional"] = float(notional)
    if reduce_only and not dry_run:
        kwargs["reduce_only"] = True
    if client_order_id:
        kwargs["client_order_id"] = client_order_id
    try:
        response = client.place_order(runtime.symbol, side, size, **kwargs)
    except TypeError as exc:
        if client_order_id and "client_order_id" in str(exc):
            raise BinanceAPIError("Signed live orders require client_order_id support") from exc
        raise
    except BinanceAPIError:
        if not dry_run and client_order_id and hasattr(client, "get_order"):
            response = client.get_order(runtime.symbol, orig_client_order_id=client_order_id)
        else:
            raise
    if dry_run:
        print("paper order:", json.dumps(response, indent=2))
        return response
    print(f"live order: {side} {size:.8f} {runtime.symbol}")
    print(json.dumps(response, indent=2))
    return response


def _order_fill_details(
    order: object,
    *,
    fallback_qty: float,
    fallback_price: float,
    allow_quantity_fallback: bool = True,
) -> tuple[float, float, float]:
    qty = 0.0
    quote = 0.0
    average = 0.0
    if isinstance(order, dict):
        fills = order.get("fills")
        if isinstance(fills, list):
            for fill in fills:
                if not isinstance(fill, dict):
                    continue
                fill_qty = _safe_float(fill.get("qty"))
                fill_price = _safe_float(fill.get("price"))
                if fill_qty <= 0.0 or fill_price <= 0.0:
                    continue
                qty += fill_qty
                quote += fill_qty * fill_price
        if qty <= 0.0:
            qty = _safe_float(order.get("executedQty"))
            if qty <= 0.0 and allow_quantity_fallback:
                qty = _safe_float(order.get("origQty"))
        quote = quote or _safe_float(
            order.get("cummulativeQuoteQty")
            or order.get("cumQuote")
            or order.get("cumBase")
            or order.get("notional")
        )
        average = _safe_float(order.get("avgPrice") or order.get("price"))
    if qty <= 0.0 and allow_quantity_fallback:
        qty = max(0.0, float(fallback_qty))
    if average <= 0.0:
        average = quote / qty if qty > 0.0 and quote > 0.0 else max(0.0, float(fallback_price))
    notional = quote if quote > 0.0 else qty * average
    return float(qty), float(average), float(notional)


def _order_query_keys(order: object) -> tuple[object | None, str | None]:
    if not isinstance(order, dict):
        return None, None
    order_id = order.get("orderId")
    client_order_id = order.get("clientOrderId") or order.get("origClientOrderId")
    return order_id, str(client_order_id) if client_order_id else None


def _order_response_text(order: object, *names: str) -> str:
    if not isinstance(order, Mapping):
        return ""
    for name in names:
        value = order.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolved_order_fill_details(
    client: BinanceClient,
    runtime: RuntimeConfig,
    order: object,
    *,
    fallback_qty: float,
    fallback_price: float,
    dry_run: bool,
) -> tuple[float, float, float, str]:
    allow_fallback = bool(dry_run)
    qty, average, notional = _order_fill_details(
        order,
        fallback_qty=fallback_qty,
        fallback_price=fallback_price,
        allow_quantity_fallback=allow_fallback,
    )
    if dry_run or qty > 0.0:
        return qty, average, notional, "order_response"
    order_id, client_order_id = _order_query_keys(order)
    if order_id is None and client_order_id is None:
        return qty, average, notional, "unresolved_no_order_id"
    if not hasattr(client, "get_order"):
        return qty, average, notional, "unresolved_no_order_query"
    try:
        refreshed = client.get_order(
            runtime.symbol,
            order_id=order_id,
            orig_client_order_id=client_order_id,
        )
    except BinanceAPIError:
        raise
    refreshed_qty, refreshed_average, refreshed_notional = _order_fill_details(
        refreshed,
        fallback_qty=0.0,
        fallback_price=fallback_price,
        allow_quantity_fallback=False,
    )
    return refreshed_qty, refreshed_average, refreshed_notional, "order_query"


def _asset_free_balance(account: object, asset: str) -> float:
    if not isinstance(account, dict):
        return 0.0
    total = 0.0
    for item in account.get("balances", []) or []:
        if not isinstance(item, dict) or item.get("asset") != asset:
            continue
        total += _safe_float(item.get("free"))
    return total


def _order_executed_qty(order: object) -> float:
    if not isinstance(order, dict):
        return 0.0
    fills = order.get("fills")
    if isinstance(fills, list):
        filled = 0.0
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            filled += max(0.0, _safe_float(fill.get("qty")))
        if filled > 0.0:
            return filled
    return _safe_float(order.get("executedQty"))


def _roundtrip_quantity(client, symbol: str, requested: float, price: float) -> tuple[float, SymbolConstraints, float]:
    if requested <= 0.0:
        raise ValueError("Roundtrip quantity must be > 0.")
    if price <= 0.0:
        raise BinanceAPIError(f"Cannot resolve a positive {symbol} mark price")
    quantity, constraints = client.normalize_quantity(symbol, requested)
    min_notional = float(constraints.min_notional)
    if min_notional > 0.0 and quantity * price < min_notional:
        step = max(0.0, _safe_float(getattr(constraints, "step_size", 0.0)))
        target = (min_notional / price) + (2.0 * step)
        quantity, constraints = client.normalize_quantity(symbol, max(target, float(constraints.min_qty)))
    notional = quantity * price
    if quantity <= 0.0:
        raise BinanceAPIError(f"Requested quantity is below {symbol} exchange filters")
    if min_notional > 0.0 and notional < min_notional:
        raise BinanceAPIError(
            f"Requested quantity notional {notional:.2f} is below exchange minimum {min_notional:.2f}"
        )
    if constraints.max_notional > 0.0 and notional > constraints.max_notional:
        raise BinanceAPIError(
            f"Requested quantity notional {notional:.2f} exceeds exchange maximum {constraints.max_notional:.2f}"
        )
    return quantity, constraints, notional


def _roundtrip_second_quantity(
    client,
    symbol: str,
    quote_asset: str,
    base_asset: str,
    side: str,
    executed_qty: float,
    account: object,
    price: float,
) -> float:
    if side == "SELL":
        available = _asset_free_balance(account, base_asset)
        target = min(max(0.0, executed_qty), available)
    else:
        available_quote = _asset_free_balance(account, quote_asset)
        target = min(max(0.0, executed_qty), (available_quote / price) * 0.995 if price > 0.0 else 0.0)
    quantity, _constraints = client.normalize_quantity(symbol, target)
    return quantity


def command_spot_roundtrip(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    if not getattr(args, "yes", False):
        print("Pass --yes to confirm signed spot testnet/demo order placement.", file=sys.stderr)
        return 2
    if runtime.market_type != "spot":
        print("Test order requires market_type=spot in Connection settings.", file=sys.stderr)
        return 2
    if not _allows_signed_execution(runtime):
        print("Test order requires testnet=true or demo=true in Connection settings.", file=sys.stderr)
        return 2
    if not _has_api_credentials(runtime):
        print(_credential_required_message("Test order"), file=sys.stderr)
        return 2

    quote_asset, base_asset = _fund_asset_labels(runtime)
    mode = str(getattr(args, "mode", "auto"))
    quantity_requested = float(getattr(args, "quantity", 0.0))
    client = _build_client(runtime)
    price = 0.0
    quantity = 0.0
    second_quantity = 0.0
    notional = 0.0
    before: object | None = None
    mid: object | None = None
    first: object | None = None
    first_side = ""
    second_side = ""
    first_fill_source = ""
    try:
        _ensure_runtime_symbol(runtime, client)
        price, _timestamp = client.get_symbol_price(runtime.symbol)
        quantity, _constraints, notional = _roundtrip_quantity(
            client,
            runtime.symbol,
            quantity_requested,
            float(price),
        )
        before = client.get_account()
        base_free = _asset_free_balance(before, base_asset)
        quote_free = _asset_free_balance(before, quote_asset)
        if mode == "auto":
            mode = "buy-sell" if quote_free >= notional * 1.01 else "sell-buy"
        if mode == "buy-sell":
            if quote_free < notional * 1.01:
                raise BinanceAPIError(
                    f"Insufficient {quote_asset} for BUY leg: need about {notional:.2f}, have {quote_free:.2f}"
                )
            first_side, second_side = "BUY", "SELL"
        elif mode == "sell-buy":
            if base_free < quantity:
                raise BinanceAPIError(
                    f"Insufficient {base_asset} for SELL leg: need {quantity:.8f}, have {base_free:.8f}"
                )
            first_side, second_side = "SELL", "BUY"
        else:
            raise ValueError(f"Unsupported roundtrip mode: {mode}")

        first = client.place_order(runtime.symbol, first_side, quantity, dry_run=False)
        executed, _first_fill_price, _first_notional, first_fill_source = _resolved_order_fill_details(
            client,
            runtime,
            first,
            fallback_qty=0.0,
            fallback_price=float(price),
            dry_run=False,
        )
        if executed <= 0.0:
            raise BinanceAPIError("First roundtrip order response did not include executed quantity")
        mid = client.get_account()
        second_quantity = _roundtrip_second_quantity(
            client,
            runtime.symbol,
            quote_asset,
            base_asset,
            second_side,
            executed,
            mid,
            float(price),
        )
        if second_quantity <= 0.0:
            raise BinanceAPIError(f"Could not size {second_side} leg from post-{first_side} balances")
        second = client.place_order(runtime.symbol, second_side, second_quantity, dry_run=False)
        after = client.get_account()
    except (BinanceAPIError, ValueError) as exc:
        if first is not None:
            partial_payload = {
                "command": "spot-roundtrip",
                "status": "partial_failed",
                "timestamp": int(time.time()),
                "runtime": _public_runtime_payload(runtime),
                "symbol": runtime.symbol,
                "mode": mode,
                "error": str(exc),
                "price_reference": float(price),
                "quantity_requested": float(quantity_requested),
                "quantity_first": float(quantity),
                "quantity_second_attempted": float(second_quantity),
                "notional_reference": float(notional),
                "balances_before": (
                    {
                        base_asset: _asset_free_balance(before, base_asset),
                        quote_asset: _asset_free_balance(before, quote_asset),
                    }
                    if before is not None
                    else None
                ),
                "balances_after_first": (
                    {
                        base_asset: _asset_free_balance(mid, base_asset),
                        quote_asset: _asset_free_balance(mid, quote_asset),
                    }
                    if mid is not None
                    else None
                ),
                "first_order": {
                    "side": first_side,
                    "status": first.get("status") if isinstance(first, dict) else None,
                    "orderId": first.get("orderId") if isinstance(first, dict) else None,
                    "executedQty": first.get("executedQty") if isinstance(first, dict) else None,
                    "fill_source": first_fill_source,
                },
                "second_order": {"side": second_side, "status": "not_completed"},
            }
            try:
                artifact_path = _persist_run_artifact("spot_roundtrip_partial", Path("data"), partial_payload)
                print(f"Partial spot roundtrip recorded at {artifact_path}", file=sys.stderr)
            except (OSError, RuntimeError, TypeError, ValueError) as persist_exc:
                print(f"Partial spot roundtrip could not be recorded: {persist_exc}", file=sys.stderr)
        print(_credential_failure_message("Test order", exc), file=sys.stderr)
        return 2

    payload = {
        "command": "spot-roundtrip",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "symbol": runtime.symbol,
        "mode": mode,
        "price_reference": float(price),
        "quantity_requested": float(quantity_requested),
        "quantity_first": float(quantity),
        "quantity_second": float(second_quantity),
        "notional_reference": float(notional),
        "balances_before": {
            base_asset: _asset_free_balance(before, base_asset),
            quote_asset: _asset_free_balance(before, quote_asset),
        },
        "balances_after": {
            base_asset: _asset_free_balance(after, base_asset),
            quote_asset: _asset_free_balance(after, quote_asset),
        },
        "first_order": {
            "side": first_side,
            "status": first.get("status") if isinstance(first, dict) else None,
            "orderId": first.get("orderId") if isinstance(first, dict) else None,
            "executedQty": first.get("executedQty") if isinstance(first, dict) else None,
            "fill_source": first_fill_source,
        },
        "second_order": {
            "side": second_side,
            "status": second.get("status") if isinstance(second, dict) else None,
            "orderId": second.get("orderId") if isinstance(second, dict) else None,
            "executedQty": second.get("executedQty") if isinstance(second, dict) else None,
        },
    }
    artifact_path = _persist_run_artifact("spot_roundtrip", Path("data"), payload)
    print(f"Spot {_runtime_environment(runtime)} roundtrip complete.")
    print(json.dumps({**payload, "artifact": str(artifact_path)}, indent=2))
    return 0


def command_configure(_: argparse.Namespace) -> int:
    current = load_runtime()
    next_runtime = prompt_runtime(current)
    save_runtime(next_runtime)

    if next_runtime.validate_account and _has_api_credentials(next_runtime):
        client = _build_client(next_runtime)
        try:
            _validate_runtime_connection(next_runtime, client)
        except BinanceAPIError as exc:
            print(f"Configuration saved, but validation failed: {exc}", file=sys.stderr)
            return 2

    print("Runtime config saved to", config_paths()["runtime"])
    print(
        f"market={next_runtime.market_type} environment={_runtime_environment(next_runtime)} "
        f"testnet={next_runtime.testnet} demo={getattr(next_runtime, 'demo', False)} paper={next_runtime.dry_run}"
    )
    if next_runtime.market_type == "futures":
        print("futures-mode enabled; leverage can be set via strategy.leverage")
    return 0


def command_connect(_: argparse.Namespace) -> int:
    runtime = load_runtime()
    if not _has_api_credentials(runtime):
        print(_credential_required_message("Connect"), file=sys.stderr)
        return 2
    if not _allows_signed_execution(runtime):
        print(
            "Connect validates signed credentials only on Binance testnet or demo. "
            "Enable testnet=true or demo=true before authenticating.",
            file=sys.stderr,
        )
        return 2
    try:
        client = _build_client(runtime)
        server_time = client.get_exchange_time()
        _ensure_runtime_symbol(runtime, client)
    except BinanceAPIError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 2
    try:
        account = client.get_account()
    except BinanceAPIError as exc:
        print(_credential_failure_message("Connect", exc), file=sys.stderr)
        return 2
    if isinstance(account, dict):
        account = {
            "updateTime": account.get("updateTime"),
            "canTrade": account.get("canTrade"),
            "accountType": account.get("accountType"),
            "positions": account.get("positions"),
            "assets": account.get("assets"),
        }
    if runtime.market_type == "futures":
        try:
            max_leverage = client.get_max_leverage(runtime.symbol)
        except BinanceAPIError as exc:
            print(f"unable to fetch leverage bracket: {exc}", file=sys.stderr)
        else:
            print(f"max leverage on {runtime.symbol}: {max_leverage}x")

    print("exchange: connected")
    print("market:", runtime.market_type)
    print("environment:", _runtime_environment(runtime))
    print("testnet:", runtime.testnet)
    print("demo:", getattr(runtime, "demo", False))
    print("endpoint:", client.base_url)
    print("server_time:", server_time.get("serverTime") if isinstance(server_time, dict) else server_time)
    print("account:", json.dumps(account, indent=2))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    ok, lines = _readiness_report(input_path=args.input, model_path=args.model, online=bool(args.online))
    print("Readiness report")
    for line in lines:
        print(line)
    return 0 if ok else 2


def command_audit(args: argparse.Namespace) -> int:
    from .audit import build_audit_report, render_audit_report

    runtime = load_runtime()
    strategy = load_strategy()
    candles = _load_rows_for_command(args.input, label="Audit data load failed")
    if candles is None:
        return 2
    report = build_audit_report(candles, runtime, strategy, model_path=Path(args.model))
    print(render_audit_report(report))
    return 0 if report.ok else 2


def command_risk(args: argparse.Namespace) -> int:
    return risk_workflows.command_risk(
        args,
        load_runtime_fn=load_runtime,
        load_strategy_fn=load_strategy,
    )


def command_reconcile(args: argparse.Namespace) -> int:
    from .positions import PositionsStore
    from .reconciliation import reconcile_account_positions
    from .storage import write_json_atomic

    runtime = load_runtime()
    if not _has_api_credentials(runtime):
        print(_credential_required_message("Reconciliation"), file=sys.stderr)
        return 2
    try:
        account = _build_client(runtime).get_account()
        report = reconcile_account_positions(
            account,
            runtime,
            PositionsStore(),
            quantity_tolerance=max(0.0, float(getattr(args, "quantity_tolerance", 1e-8))),
        )
    except (BinanceAPIError, OSError, ValueError) as exc:
        print(f"reconcile failed: {exc}", file=sys.stderr)
        return 2
    output = Path(getattr(args, "output", "data/autonomous/reconciliation.json"))
    write_json_atomic(output, report.asdict(), indent=2, sort_keys=True)
    if getattr(args, "json", False):
        print(json.dumps(report.asdict(), indent=2))
    else:
        status = "ok" if report.ok else "mismatch"
        print(
            f"reconcile: {status} market={report.market_type} "
            f"local_live={report.local_live_open_count} paper={report.local_paper_open_count} "
            f"exchange={report.exchange_exposure_count}"
        )
        for mismatch in report.mismatches:
            print(
                f"  {mismatch.reason}: {mismatch.symbol} {mismatch.side} "
                f"local={mismatch.local_qty:.8f} exchange={mismatch.exchange_qty:.8f} "
                f"diff={mismatch.difference:+.8f}"
            )
        for warning in report.warnings:
            print(f"  warning: {warning}")
        print(f"  report -> {output}")
    return 0 if report.ok else 2


def command_universe(args: argparse.Namespace) -> int:
    from .assets import normalize_symbols
    from .market_universe import select_tradeable_universe

    runtime = load_runtime()
    strategy = load_strategy()
    requested = (
        normalize_symbols(str(args.symbols).split(","))
        if getattr(args, "symbols", None)
        else tuple(runtime.symbols)
    )
    try:
        selection = select_tradeable_universe(
            _build_client(runtime),
            requested,
            strategy,
            quote_asset=runtime.quote_asset,
        )
    except BinanceAPIError as exc:
        print(f"universe selection failed: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(selection.asdict(), indent=2))
    else:
        print(
            f"universe quote={selection.quote_asset} allowed={selection.allowed} "
            f"eligible={len(selection.eligible)}/{len(selection.requested)} min_required={selection.min_required}"
        )
        for item in selection.eligible:
            print(
                f"ok {item.symbol}: volume={item.quote_volume:.0f} trades={item.trade_count} "
                f"spread={item.spread_bps:.2f}bps score={item.liquidity_score:.2f}"
            )
        for item in selection.rejected:
            print(
                f"reject {item.symbol}: volume={item.quote_volume:.0f} trades={item.trade_count} "
                f"spread={item.spread_bps:.2f}bps score={item.liquidity_score:.2f} "
                f"reasons={','.join(item.reasons)}"
            )
    return 0 if selection.allowed else 2


def command_report(args: argparse.Namespace) -> int:
    if bool(args.account) and not _has_api_credentials(load_runtime()):
        print(_credential_required_message("Full report account section"), file=sys.stderr)
        return 2
    report = _render_operator_report(
        with_account=bool(args.account),
        doctor=bool(args.doctor),
        online=bool(args.online),
        input_path=args.input,
        model_path=args.model,
    )
    print(report)
    if bool(args.account) and "Account balances failed:" in report:
        return 2
    return 0


def command_coordinator(args: argparse.Namespace) -> int:
    from .coordinator import build_runtime_coordinator, render_coordinator_decision

    decision = build_runtime_coordinator(
        load_runtime(),
        load_strategy(),
        model_path=Path(getattr(args, "model", "data/model.json")),
        positions_root=Path(getattr(args, "positions_root", "data/autonomous")),
    )
    if getattr(args, "json", False):
        print(json.dumps(decision.asdict(), indent=2, sort_keys=True))
    else:
        print(render_coordinator_decision(decision))
    return 0


def command_prepare(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    try:
        runtime = load_runtime()
        max_batch_size = 1500 if runtime.market_type == "futures" else 1000
        preset = _parse_training_preset(str(args.preset))
        limit = int(args.limit)
        batch_size = int(getattr(args, "batch_size", 1000))
        seed = int(args.seed)
        learning_rate = float(getattr(args, "learning_rate", 0.05))
        l2_penalty = float(getattr(args, "l2_penalty", 1e-4))
        start_cash = float(args.start_cash)
        preset_args = _apply_training_preset(
            argparse.Namespace(
                preset=preset,
                epochs=250,
                walk_forward=False,
                walk_forward_train=300,
                walk_forward_test=60,
                walk_forward_step=30,
                calibrate_threshold=False,
            )
        )
        epochs = int(args.epochs if getattr(args, "epochs", None) is not None else preset_args.epochs)
        walk_forward = (
            bool(args.walk_forward)
            if getattr(args, "walk_forward", None) is not None
            else bool(preset_args.walk_forward)
        )
        walk_forward_train = int(
            args.walk_forward_train
            if getattr(args, "walk_forward_train", None) is not None
            else preset_args.walk_forward_train
        )
        walk_forward_test = int(
            args.walk_forward_test
            if getattr(args, "walk_forward_test", None) is not None
            else preset_args.walk_forward_test
        )
        walk_forward_step = int(
            args.walk_forward_step
            if getattr(args, "walk_forward_step", None) is not None
            else preset_args.walk_forward_step
        )
        should_calibrate_threshold = (
            bool(args.calibrate_threshold)
            if getattr(args, "calibrate_threshold", None) is not None
            else bool(preset_args.calibrate_threshold)
        )
        if limit < 1:
            raise ValueError("Fetch limit must be >= 1.")
        if batch_size < 1:
            raise ValueError("Fetch batch size must be >= 1.")
        batch_size = min(batch_size, max_batch_size)
        if epochs < 1:
            raise ValueError("Training epochs must be >= 1.")
        if seed < 0:
            raise ValueError("Training seed must be >= 0.")
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be > 0.")
        if l2_penalty < 0.0:
            raise ValueError("L2 penalty must be >= 0.")
        if start_cash < 1:
            raise ValueError("Backtest starting cash must be >= 1.")
        if walk_forward_train < 2:
            raise ValueError("Walk-forward train window must be >= 2.")
        if walk_forward_test < 1:
            raise ValueError("Walk-forward test window must be >= 1.")
        if walk_forward_step < 1:
            raise ValueError("Walk-forward step must be >= 1.")
    except (TypeError, ValueError) as exc:
        print(f"Prepare settings invalid: {exc}", file=sys.stderr)
        return 2

    historical = str(args.historical)
    model = str(args.model)
    sequence = [
        (
            "Download market data",
            command_fetch,
            argparse.Namespace(symbol=runtime.symbol, interval=runtime.interval, limit=limit, batch_size=batch_size, output=historical),
        ),
        (
            "Train AI model",
            command_train,
            argparse.Namespace(
                input=historical,
                output=model,
                preset="custom",
                requested_preset=preset,
                epochs=epochs,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                seed=seed,
                walk_forward=walk_forward,
                walk_forward_train=walk_forward_train,
                walk_forward_test=walk_forward_test,
                walk_forward_step=walk_forward_step,
                calibrate_threshold=should_calibrate_threshold,
            ),
        ),
        (
            "Evaluate model",
            command_evaluate,
            argparse.Namespace(input=historical, model=model, threshold=None, calibrate_threshold=should_calibrate_threshold),
        ),
        (
            "Backtest strategy",
            command_backtest,
            argparse.Namespace(input=historical, model=model, start_cash=start_cash),
        ),
        (
            "Data/model audit",
            command_audit,
            argparse.Namespace(input=historical, model=model),
        ),
        (
            "Readiness check",
            command_doctor,
            argparse.Namespace(input=historical, model=model, online=bool(args.online_doctor)),
        ),
    ]
    for label, fn, step_args in sequence:
        print(f"== {label} ==")
        result = fn(step_args)
        if result != 0:
            print(f"Prepare stopped at {label}.", file=sys.stderr)
            return result
    return 0


def command_status(args: argparse.Namespace) -> int:
    from .autonomous import AutonomousControl
    from .positions import PositionsStore

    runtime = load_runtime()
    strategy = load_strategy()
    if bool(getattr(args, "compact", False)):
        if runtime.testnet:
            environment = "testnet"
        elif runtime.demo:
            environment = "demo"
        elif runtime.dry_run:
            environment = "paper"
        else:
            environment = "non-mainnet"
        execution = "paper" if runtime.dry_run else "live"
        state = str(AutonomousControl().read().get("state") or "UNKNOWN").lower()
        position_store = PositionsStore()
        ledger_errors = position_store.open_integrity_errors()
        position_count = len(position_store.load_open()) if not ledger_errors else 0
        ledger_state = "invalid" if ledger_errors else ("tracked" if position_count else "clear")
        print(
            f"environment={environment} bot_state={state} risk={strategy.risk_level} "
            f"leverage={strategy.leverage:g} ai={'enabled' if runtime.ai_enabled else 'disabled'} "
            f"reinvest={'on' if strategy.reinvest_profits else 'off'} symbol={runtime.symbol} "
            f"market={runtime.market_type} execution={execution} positions={position_count} "
            f"ledger={ledger_state}"
        )
        return 0
    print(json.dumps({"runtime": _public_runtime_payload(runtime), "strategy": strategy.asdict()}, indent=2))
    return 0


def command_compute(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    requested = str(getattr(args, "backend", None) or runtime.compute_backend or default_compute_backend()).lower()
    if requested not in _COMPUTE_BACKEND_CHOICES:
        print(f"Unknown compute backend {requested!r}.", file=sys.stderr)
        return 2
    info = resolve_backend(requested)
    if getattr(args, "backend", None) is not None:
        runtime.compute_backend = requested
        if info.kind == "cpu" and runtime.ai_enabled:
            runtime.ai_enabled = False
            print("AI features disabled because the selected compute backend is CPU-only.", file=sys.stderr)
        save_runtime(runtime)
    print(describe_backend(info))
    if info.kind == "cpu":
        print(
            "warning: CPU-only mode is allowed, but training/backtesting will be slower and AI features cannot run.",
            file=sys.stderr,
        )
    return 0


def command_ai(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    enable_requested = bool(getattr(args, "enable", False))
    if bool(getattr(args, "enable", False)) and bool(getattr(args, "disable", False)):
        print("--enable and --disable cannot be combined.", file=sys.stderr)
        return 2
    if bool(getattr(args, "require_gpu", False)) and bool(getattr(args, "no_require_gpu", False)):
        print("--require-gpu and --no-require-gpu cannot be combined.", file=sys.stderr)
        return 2
    if bool(getattr(args, "allow_paper_fallback", False)) and bool(getattr(args, "no_paper_fallback", False)):
        print("--allow-paper-fallback and --no-paper-fallback cannot be combined.", file=sys.stderr)
        return 2

    changed = False
    if enable_requested:
        runtime.ai_enabled = True
        changed = True
    if bool(getattr(args, "disable", False)):
        runtime.ai_enabled = False
        changed = True
    if getattr(args, "provider", None):
        runtime.ai_provider = str(args.provider)
        changed = True
    if getattr(args, "model", None):
        runtime.ai_model = str(args.model)
        changed = True
    if bool(getattr(args, "require_gpu", False)):
        runtime.ai_require_gpu = True
        changed = True
    if bool(getattr(args, "no_require_gpu", False)):
        runtime.ai_require_gpu = False
        changed = True
    if getattr(args, "min_free_vram_gb", None) is not None:
        runtime.ai_min_free_vram_gb = max(0.0, float(args.min_free_vram_gb))
        changed = True
    if getattr(args, "min_free_ram_gb", None) is not None:
        runtime.ai_min_free_ram_gb = max(0.0, float(args.min_free_ram_gb))
        changed = True
    if getattr(args, "min_model_parameters_b", None) is not None:
        runtime.ai_min_model_parameters_b = max(0.0, float(args.min_model_parameters_b))
        changed = True
    if bool(getattr(args, "allow_paper_fallback", False)):
        runtime.ai_allow_paper_fallback = True
        changed = True
    if bool(getattr(args, "no_paper_fallback", False)):
        runtime.ai_allow_paper_fallback = False
        changed = True
    if changed:
        save_runtime(runtime)

    report = detect_ai_capabilities(runtime.ai_runtime_config())
    enable_blocked = runtime.ai_enabled and report.compute_backend_kind == "cpu"
    if runtime.ai_enabled and report.compute_backend_kind == "cpu":
        runtime.ai_enabled = False
        save_runtime(runtime)
        report = detect_ai_capabilities(runtime.ai_runtime_config())
        print(
            "AI features were disabled because the selected compute backend resolved to CPU-only. "
            "Install torch-directml or choose a GPU backend, then re-enable AI.",
            file=sys.stderr,
        )
    if getattr(args, "json", False):
        print(json.dumps({"runtime_ai": runtime.ai_runtime_config().asdict(), "capabilities": report.asdict()}, indent=2))
    else:
        if changed:
            print("Saved AI runtime settings.")
        print(render_ai_capability_report(report))
    if enable_requested and enable_blocked:
        return 2
    return 0 if (report.ok or not runtime.ai_enabled or runtime.ai_allow_paper_fallback) else 2


def command_ai_benchmark(args: argparse.Namespace) -> int:
    from .ai_model_benchmark import benchmark_finance_ai_models, write_benchmark_report

    raw_models = str(getattr(args, "models", "") or "")
    models = [item.strip() for item in raw_models.split(",") if item.strip()]
    json_mode = bool(getattr(args, "json", False))

    def progress(phase: str, payload: Mapping[str, object]) -> None:
        if json_mode:
            return
        if phase == "model_started":
            print(
                f"ai-benchmark model {payload['model_index']}/{payload['model_count']}: "
                f"{payload['model']} ({payload['case_count']} cases)",
                flush=True,
            )
        elif phase == "case_complete":
            match = "match" if payload["action_match"] else "mismatch"
            print(
                f"  case {payload['case_index']}/{payload['case_count']} "
                f"{payload['case']}: {match} "
                f"latency={float(payload['latency_seconds']):.2f}s",
                flush=True,
            )

    try:
        report = benchmark_finance_ai_models(
            models=models,
            base_url=str(getattr(args, "url", None) or "http://127.0.0.1:11434"),
            timeout_seconds=max(0.1, float(getattr(args, "timeout", 20.0))),
            minimum_score=max(0.0, min(1.0, float(getattr(args, "minimum_score", 0.78)))),
            progress=progress,
        )
        output = write_benchmark_report(report, Path(getattr(args, "output", "data/ai_model_benchmark.json")))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ai-benchmark failed: {exc}", file=sys.stderr)
        return 2
    if json_mode:
        print(json.dumps(report.asdict(), indent=2, sort_keys=True))
    else:
        print(
            "ai-benchmark: "
            f"passed={report.passed} selected={report.selected_model or 'none'} "
            f"models={len(report.results)}"
        )
        for result in report.results:
            status = "pass" if result.passed else "fail"
            params = f"{result.model_parameters_b:.1f}B" if result.model_parameters_b is not None else "unknown"
            print(
                f"  {status:<4} {result.model:<24} score={result.score:.3f} "
                f"actions={result.action_match_cases}/{len(report.tests)} "
                f"json={result.valid_json_cases}/{len(report.tests)} "
                f"params={params} avg_latency={result.average_latency_seconds:.2f}s"
            )
            if result.failures:
                print(f"       first_failure={result.failures[0]}")
        print(f"  benchmark -> {output}")
    return 0 if report.passed else 2


def command_ai_forecast_benchmark(args: argparse.Namespace) -> int:
    from .foundation_benchmark import (
        FoundationBenchmarkConfig,
        parse_utc_ms,
        run_foundation_forecast_benchmark,
    )

    try:
        config = FoundationBenchmarkConfig(
            database_path=str(args.database),
            source_cache_root=(str(args.source_cache) if args.source_cache else None),
            model_size=str(args.model_size),
            backend=str(args.backend),
            bootstrap_source=bool(args.bootstrap_source or args.repair_source),
            repair_source=bool(args.repair_source),
            require_accelerator=not bool(args.allow_cpu),
            start_ms=parse_utc_ms(str(args.start)),
            end_exclusive_ms=parse_utc_ms(str(args.end_exclusive)),
            samples_per_symbol=int(args.samples_per_symbol),
            lookback_bars=int(args.lookback_bars),
            prediction_bars=int(args.prediction_bars),
            batch_size=int(args.batch_size),
            inference_samples=int(args.inference_samples),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            include_volume=bool(args.include_volume),
            seed=int(args.seed),
            bootstrap_samples=int(args.bootstrap_samples),
            worker_timeout_seconds=float(args.worker_timeout),
            max_worker_restarts=int(args.max_worker_restarts),
            worker_rotation_batches=int(args.worker_rotation_batches),
        )
        report = run_foundation_forecast_benchmark(
            config,
            observations_path=Path(args.observations),
            chart_path=Path(args.chart),
            report_path=Path(args.output),
            progress=lambda message: print(f"ai-forecast-benchmark: {message}", file=sys.stderr),
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ai-forecast-benchmark failed: {exc}", file=sys.stderr)
        return 2
    if bool(args.json):
        print(json.dumps(report.asdict(), indent=2, sort_keys=True))
    else:
        overall = report.metrics["overall"]
        print(
            "ai-forecast-benchmark: "
            f"status={report.status} observations={report.observation_count} "
            f"mae={float(overall['model_mae']):.8f} "
            f"random_walk_mae={float(overall['random_walk_mae']):.8f} "
            f"ic={float(overall['information_coefficient']):.4f} "
            f"direction={float(overall['direction_accuracy']):.3f}"
        )
        if report.reasons:
            print(f"  first_rejection={report.reasons[0]}")
        print(f"  observations -> {report.observations_path}")
        print(f"  chart -> {report.chart_path}")
        print(f"  report -> {args.output}")
        print("  trading_authority=false; no after-cost or order-placement claim")
    return 0 if report.predictive_candidate else 2


def command_ai_review(args: argparse.Namespace) -> int:
    from .ai_review import run_model_lab_ai_review

    runtime = load_runtime()
    try:
        review = run_model_lab_ai_review(
            Path(args.report),
            runtime,
            model=getattr(args, "model", None),
            base_url=str(getattr(args, "url", None) or "http://127.0.0.1:11434"),
            timeout_seconds=max(0.1, float(getattr(args, "timeout", 20.0))),
            output_path=(Path(args.output) if getattr(args, "output", None) else None),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ai-review failed: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(review.asdict(), indent=2))
    else:
        decision = review.decision
        print(
            f"ai-review: status={review.status} action={decision.action} "
            f"approved={review.approved} risk={decision.risk_score:.2f} "
            f"confidence={decision.confidence:.2f}"
        )
        print(f"  rationale: {decision.rationale}")
        if decision.concerns:
            print("  concerns:")
            for concern in decision.concerns:
                print(f"    - {concern}")
        if decision.required_actions:
            print("  required actions:")
            for action in decision.required_actions:
                print(f"    - {action}")
        if review.error:
            print(f"  error: {review.error}")
        if review.output_path:
            print(f"  review -> {review.output_path}")
    return 0 if review.approved else 2


def command_strategy(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    cfg = load_strategy()
    runtime = load_runtime()
    try:
        profile = _parse_strategy_profile(str(getattr(args, "profile", "custom") or "custom"))
    except ValueError as exc:
        print(f"Invalid strategy profile: {exc}", file=sys.stderr)
        return 2
    updates = dict(_STRATEGY_PROFILES[profile])
    if getattr(args, "risk_level", None) is not None:
        updates["risk_level"] = str(args.risk_level)
    if bool(getattr(args, "reinvest_profits", False)):
        updates["reinvest_profits"] = True
    if bool(getattr(args, "no_reinvest_profits", False)):
        updates["reinvest_profits"] = False
    if args.leverage is not None:
        requested = max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, args.leverage))
        if runtime.market_type == "futures":
            if runtime.api_key and runtime.api_secret:
                try:
                    client = _build_client(runtime)
                    max_leverage = client.get_max_leverage(runtime.symbol)
                except BinanceAPIError:
                    max_leverage = MAX_AUTONOMOUS_LEVERAGE
            else:
                max_leverage = MAX_AUTONOMOUS_LEVERAGE
            requested = min(requested, float(max_leverage))
        updates["leverage"] = requested
    if args.risk is not None:
        updates["risk_per_trade"] = max(0.0001, args.risk)
    if args.max_position is not None:
        updates["max_position_pct"] = max(0.0, min(1.0, args.max_position))
    if args.stop is not None:
        updates["stop_loss_pct"] = max(0.0, min(0.99, args.stop))
    if args.take is not None:
        updates["take_profit_pct"] = max(0.0, min(0.99, args.take))
    if args.max_open is not None:
        updates["max_open_positions"] = max(0, args.max_open)
    if getattr(args, "min_diversified_assets", None) is not None:
        updates["min_diversified_assets"] = max(1, int(args.min_diversified_assets))
    if getattr(args, "max_asset_allocation", None) is not None:
        updates["max_asset_allocation_pct"] = _clamp(float(args.max_asset_allocation), 0.01, 1.0)
    if getattr(args, "max_portfolio_risk", None) is not None:
        updates["max_portfolio_risk_pct"] = _clamp(float(args.max_portfolio_risk), 0.0, 1.0)
    if getattr(args, "min_quote_volume_usdc", None) is not None:
        updates["min_quote_volume_usdc"] = max(0.0, float(args.min_quote_volume_usdc))
    if getattr(args, "min_trade_count_24h", None) is not None:
        updates["min_trade_count_24h"] = max(0, int(args.min_trade_count_24h))
    if getattr(args, "max_spread_bps", None) is not None:
        updates["max_spread_bps"] = max(0.0, float(args.max_spread_bps))
    if getattr(args, "min_liquidity_score", None) is not None:
        updates["min_liquidity_score"] = _clamp(float(args.min_liquidity_score), 0.0, 1.0)
    if getattr(args, "unpredictability_cooldown", None) is not None:
        updates["unpredictability_cooldown_minutes"] = max(0, int(args.unpredictability_cooldown))
    if getattr(args, "max_regime_unpredictability", None) is not None:
        updates["max_regime_unpredictability"] = _clamp(float(args.max_regime_unpredictability), 0.0, 1.0)
    if getattr(args, "max_prediction_entropy", None) is not None:
        updates["max_prediction_entropy"] = _clamp(float(args.max_prediction_entropy), 0.0, 1.0)
    if getattr(args, "min_model_confidence", None) is not None:
        updates["min_model_confidence"] = _clamp(float(args.min_model_confidence), 0.0, 1.0)
    if args.max_trades_per_day is not None:
        updates["max_trades_per_day"] = max(0, args.max_trades_per_day)
    if args.cooldown is not None:
        updates["cooldown_minutes"] = max(0, args.cooldown)
    if getattr(args, "min_position_hold_bars", None) is not None:
        updates["min_position_hold_bars"] = max(0, int(args.min_position_hold_bars))
    if getattr(args, "flat_signal_exit_grace_bars", None) is not None:
        updates["flat_signal_exit_grace_bars"] = max(0, int(args.flat_signal_exit_grace_bars))
    if getattr(args, "max_position_hold_bars", None) is not None:
        updates["max_position_hold_bars"] = max(0, int(args.max_position_hold_bars))
    if args.signal_threshold is not None:
        updates["signal_threshold"] = _clamp(args.signal_threshold, 0.01, 0.99)
    if args.max_drawdown is not None:
        updates["max_drawdown_limit"] = max(0.0, args.max_drawdown)
    if getattr(args, "max_daily_loss", None) is not None:
        updates["max_daily_loss_pct"] = _clamp(float(args.max_daily_loss), 0.0, 0.25)
    if getattr(args, "max_session_loss", None) is not None:
        updates["max_session_loss_pct"] = _clamp(float(args.max_session_loss), 0.0, 0.50)
    if getattr(args, "max_consecutive_losses", None) is not None:
        updates["max_consecutive_losses"] = max(0, int(args.max_consecutive_losses))
    if getattr(args, "max_network_errors", None) is not None:
        updates["max_network_errors"] = max(1, int(args.max_network_errors))
    if getattr(args, "recovery_cooldown_seconds", None) is not None:
        updates["recovery_cooldown_seconds"] = max(0, int(args.recovery_cooldown_seconds))
    if args.taker_fee_bps is not None:
        updates["taker_fee_bps"] = max(0.0, args.taker_fee_bps)
    if args.slippage_bps is not None:
        updates["slippage_bps"] = max(0.0, args.slippage_bps)
    if args.label_threshold is not None:
        updates["label_threshold"] = max(0.0001, args.label_threshold)
    if getattr(args, "model_lookback", None) is not None:
        updates["model_lookback"] = max(10, int(args.model_lookback))
    if getattr(args, "training_epochs", None) is not None:
        updates["training_epochs"] = max(1, int(args.training_epochs))
    if getattr(args, "confidence_beta", None) is not None:
        updates["confidence_beta"] = _clamp(float(args.confidence_beta), 0.0, 1.0)
    if getattr(args, "external_signals", None) is not None:
        updates["external_signals_enabled"] = bool(args.external_signals)
    if getattr(args, "external_signal_max_adjustment", None) is not None:
        updates["external_signal_max_adjustment"] = _clamp(float(args.external_signal_max_adjustment), 0.0, 0.20)
    if getattr(args, "external_signal_min_providers", None) is not None:
        updates["external_signal_min_providers"] = max(0, min(120, int(args.external_signal_min_providers)))
    if getattr(args, "external_signal_ttl", None) is not None:
        updates["external_signal_ttl_seconds"] = max(0, int(args.external_signal_ttl))
    if getattr(args, "external_signal_timeout", None) is not None:
        updates["external_signal_timeout_seconds"] = _clamp(float(args.external_signal_timeout), 0.1, 30.0)
    if getattr(args, "external_news_ai", None) is not None:
        updates["external_news_ai_enabled"] = bool(args.external_news_ai)
    if getattr(args, "external_news_ai_model", None):
        updates["external_news_ai_model"] = str(args.external_news_ai_model)
    if getattr(args, "external_news_ai_url", None):
        updates["external_news_ai_url"] = str(args.external_news_ai_url)
    if getattr(args, "external_news_ai_timeout", None) is not None:
        updates["external_news_ai_timeout_seconds"] = _clamp(float(args.external_news_ai_timeout), 0.1, 30.0)
    if getattr(args, "external_news_provider_limit", None) is not None:
        updates["external_signal_news_provider_limit"] = max(0, min(120, int(args.external_news_provider_limit)))
    if getattr(args, "external_provider_parallelism", None) is not None:
        updates["external_signal_provider_parallelism"] = max(1, min(64, int(args.external_provider_parallelism)))
    if getattr(args, "external_provider_jitter", None) is not None:
        updates["external_signal_provider_jitter_seconds"] = _clamp(float(args.external_provider_jitter), 0.0, 30.0)
    if getattr(args, "external_poll_jitter", None) is not None:
        updates["external_signal_poll_jitter_seconds"] = _clamp(float(args.external_poll_jitter), 0.0, 60.0)
    if getattr(args, "telemetry_db", None):
        updates["telemetry_db_path"] = str(args.telemetry_db)
    if getattr(args, "no_telemetry", None) is not None:
        updates["telemetry_enabled"] = not bool(args.no_telemetry)
    if getattr(args, "source_grading", None) is not None:
        updates["source_grading_enabled"] = bool(args.source_grading)
    if getattr(args, "source_grading_interval", None) is not None:
        updates["source_grading_interval_seconds"] = max(60, int(args.source_grading_interval))
    if getattr(args, "source_grading_window_hours", None) is not None:
        updates["source_grading_window_hours"] = max(1, int(args.source_grading_window_hours))
    if getattr(args, "source_grade_max_age_hours", None) is not None:
        updates["source_grade_max_age_hours"] = _clamp(float(args.source_grade_max_age_hours), 0.0, 8760.0)
    feature_window_short = getattr(args, "feature_window_short", None)
    feature_window_long = getattr(args, "feature_window_long", None)
    if feature_window_short is not None or feature_window_long is not None:
        short_window = max(1, int(feature_window_short if feature_window_short is not None else cfg.feature_windows[0]))
        long_window = max(short_window + 1, int(feature_window_long if feature_window_long is not None else cfg.feature_windows[1]))
        updates["feature_windows"] = (short_window, long_window)
    try:
        if getattr(args, "set_features", None):
            updates["enabled_features"] = normalize_enabled_features(
                [part.strip() for part in str(args.set_features).split(",") if part.strip()]
            )
        else:
            selected_features = list(cfg.enabled_features)
            for name in getattr(args, "enable_feature", []) or []:
                if name not in selected_features:
                    selected_features.append(name)
            for name in getattr(args, "disable_feature", []) or []:
                selected_features = [feature for feature in selected_features if feature != name]
            if getattr(args, "enable_feature", None) or getattr(args, "disable_feature", None):
                updates["enabled_features"] = normalize_enabled_features(selected_features)
    except ValueError as exc:
        print(f"Invalid feature selection: {exc}", file=sys.stderr)
        return 2

    cfg = StrategyConfig(**{**cfg.asdict(), **updates})
    save_strategy(cfg)
    print("Saved strategy settings.")
    if cfg.reinvest_profits:
        print("WARNING: profit reinvestment is enabled; position sizing can compound losses as well as gains.")
    print(json.dumps(cfg.asdict(), indent=2))
    return 0


def _runtime_with_market(runtime: RuntimeConfig, market_type: str) -> RuntimeConfig:
    return data_workflows.runtime_with_market(runtime, market_type)


def _data_sync_config_from_args(args: argparse.Namespace, runtime: RuntimeConfig) -> MarketDataSyncConfig:
    return data_workflows.data_sync_config_from_args(args, runtime)


def _start_background_data_sync(args: argparse.Namespace) -> int:
    return data_workflows.start_background_data_sync(
        args,
        python_executable=sys.executable,
        popen=subprocess.Popen,
    )


def command_data_sync(args: argparse.Namespace) -> int:
    return data_workflows.command_data_sync(
        args,
        load_runtime_fn=load_runtime,
        build_client_fn=_build_client,
        sync_market_data_fn=sync_market_data,
        render_sync_result_fn=render_sync_result,
        sleep_fn=time.sleep,
        python_executable=sys.executable,
        popen=subprocess.Popen,
    )


def _data_health_iso(ts_ms: int | None) -> str:
    if ts_ms is None:
        return ""
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return ""


def _count_by(items: Sequence[object], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(getattr(item, attr, "") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def command_data_health(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    market_filter = getattr(args, "market", None) or runtime.market_type
    interval_filter = getattr(args, "interval", None) or runtime.interval
    raw_symbols = str(getattr(args, "symbols", "") or "").strip()
    symbols = [item.strip().upper() for item in raw_symbols.split(",") if item.strip()]
    if not symbols and getattr(args, "symbol", None):
        symbols = [str(args.symbol).upper()]
    min_rows = max(0, int(getattr(args, "min_rows", 0) or 0))
    min_coverage_ratio = max(0.0, min(1.0, float(getattr(args, "min_coverage_ratio", 0.995) or 0.995)))
    max_gap_count = max(0, int(getattr(args, "max_gap_count", 0) or 0))
    require_verified_checksum = bool(getattr(args, "require_verified_checksum", False))
    db_path = Path(getattr(args, "db", "data/market_data.sqlite"))

    try:
        step_ms = interval_milliseconds(interval_filter)
    except ValueError as exc:
        print(f"data-health failed: {exc}", file=sys.stderr)
        return 2

    items: list[dict[str, object]] = []
    with MarketDataStore(db_path) as store:
        if symbols:
            coverages = [store.coverage(symbol, market_filter, interval_filter) for symbol in symbols]
        else:
            coverages = store.candle_series(market_type=market_filter, interval=interval_filter)
        for coverage in coverages:
            quality = store.coverage_quality(coverage.symbol, coverage.market_type, coverage.interval, step_ms)
            archives = store.archive_files(symbol=coverage.symbol, market_type=coverage.market_type, interval=coverage.interval)
            archive_status_counts = _count_by(archives, "status")
            checksum_status_counts = _count_by(archives, "checksum_status")
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
            item_status = "ok" if not reasons else "block"
            items.append(
                {
                    "status": item_status,
                    "symbol": coverage.symbol,
                    "market_type": coverage.market_type,
                    "interval": coverage.interval,
                    "rows": quality.coverage.count,
                    "expected_rows": quality.expected_count,
                    "first_open_time": quality.coverage.first_open_time,
                    "last_open_time": quality.coverage.last_open_time,
                    "first_utc": _data_health_iso(quality.coverage.first_open_time),
                    "last_utc": _data_health_iso(quality.coverage.last_open_time),
                    "coverage_ratio": quality.coverage_ratio,
                    "gap_count": quality.gap_count,
                    "archive_status_counts": archive_status_counts,
                    "checksum_status_counts": checksum_status_counts,
                    "reasons": reasons,
                    "warnings": warnings,
                }
            )

    overall_status = "ok" if items and all(item["status"] == "ok" for item in items) else "block"
    payload = {
        "status": overall_status,
        "db": str(db_path),
        "market_type": market_filter,
        "interval": interval_filter,
        "symbol_count": len(items),
        "min_rows": min_rows,
        "min_coverage_ratio": min_coverage_ratio,
        "max_gap_count": max_gap_count,
        "require_verified_checksum": require_verified_checksum,
        "items": items,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "data-health: "
            f"status={overall_status} symbols={len(items)} market={market_filter} interval={interval_filter} "
            f"min_rows={min_rows} min_coverage={min_coverage_ratio:.4f} max_gaps={max_gap_count}"
        )
        for item in items:
            print(
                f"{item['symbol']} {item['status']} rows={item['rows']} "
                f"coverage={float(item['coverage_ratio']):.4f} gaps={item['gap_count']} "
                f"checksum={item['checksum_status_counts']}"
            )
            for reason in item["reasons"]:
                print(f"warning: {item['symbol']} {reason}", file=sys.stderr)
            for warning in item.get("warnings", []):
                print(f"warning: {item['symbol']} {warning}", file=sys.stderr)
    return 0 if overall_status == "ok" else 2


def command_api_budget(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    market_type = getattr(args, "market", None) or runtime.market_type
    runtime = _runtime_with_market(runtime, market_type)
    db_path = Path(getattr(args, "db", "data/market_data.sqlite"))
    report: object | None
    cached = _latest_api_budget_snapshot(db_path, market_type)
    max_age_seconds = max(60, min(120, int(getattr(args, "max_age_seconds", 90) or 90)))
    should_refresh = bool(getattr(args, "refresh", False))
    if not getattr(args, "cached_only", False) and cached is None:
        should_refresh = True
    if (
        not getattr(args, "cached_only", False)
        and cached is not None
        and not _api_budget_snapshot_is_fresh(cached, max_age_ms=max_age_seconds * 1000)
    ):
        should_refresh = True
    if should_refresh:
        try:
            client = _build_client(runtime)
            report = _refresh_api_budget_report(runtime, client, db_path=db_path)
        except (BinanceAPIError, OSError, ValueError) as exc:
            if cached is None or getattr(args, "refresh", False):
                print(f"API budget refresh failed: {exc}", file=sys.stderr)
                return 2
            print(f"API budget refresh failed; using cached sample: {exc}", file=sys.stderr)
            report = cached
    else:
        report = cached

    if getattr(args, "json", False):
        payload = report.asdict() if hasattr(report, "asdict") else (dict(report) if isinstance(report, Mapping) else {})
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if getattr(args, "compact", False):
        print(summarize_api_budget(report if isinstance(report, Mapping) else report))
        return 0
    print(render_api_budget(report if isinstance(report, Mapping) else report))
    return 0


def command_polymarket_record(args: argparse.Namespace) -> int:
    """Capture public, prospective evidence for the Polymarket paper lane."""
    try:
        database = Path(args.database)
        progress_path_value = str(getattr(args, "progress_path", None) or "").strip()
        progress_path = Path(progress_path_value) if progress_path_value else None
        if progress_path is not None and progress_path.resolve() == database.resolve():
            raise ValueError("--progress-path must not overwrite the evidence database")
        progress_write_warning_emitted = False

        def progress(_phase: str, payload: Mapping[str, object]) -> None:
            nonlocal progress_write_warning_emitted
            print(
                "polymarket-record-progress: "
                f"phase={payload.get('phase')} "
                f"elapsed={float(payload.get('elapsed_seconds', 0.0)):.1f}s "
                f"messages={int(payload.get('written_message_count', 0))} "
                f"queue={int(payload.get('queue_size', 0))} "
                f"verified_messages="
                f"{int(payload.get('verified_raw_message_count', 0))} "
                f"verified_events={int(payload.get('verified_event_count', 0))}",
                file=sys.stderr,
                flush=True,
            )
            if progress_path is None:
                return
            try:
                write_json_atomic(
                    progress_path,
                    dict(payload),
                    indent=2,
                    sort_keys=True,
                )
            except Exception as exc:
                if not progress_write_warning_emitted:
                    print(
                        "polymarket-record progress sidecar failed: "
                        f"{exc.__class__.__name__}: {exc}",
                        file=sys.stderr,
                    )
                    progress_write_warning_emitted = True

        recorder = PolymarketPublicRecorder(
            database,
            queue_capacity=int(args.queue_capacity),
            discovery_interval_seconds=int(args.discovery_interval_seconds),
            memory_limit=str(args.memory_limit),
            database_threads=int(args.database_threads),
        )
        report = asyncio.run(
            recorder.run(
                duration_seconds=int(args.duration_seconds),
                progress=progress,
                progress_interval_seconds=int(args.progress_interval_seconds),
            )
        )
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(f"polymarket-record failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(report.asdict(), indent=2, sort_keys=True))
    else:
        print(
            "polymarket-record: "
            f"status={report.status} duration={report.duration_seconds:.3f}s "
            f"markets={report.market_snapshot_count} messages={report.raw_message_count} "
            f"events={report.normalized_event_count} gaps={report.stream_gap_count}"
        )
        for stream, count in sorted(report.stream_counts.items()):
            print(f"  {stream}: {count}")
        print(f"database: {report.database}")
        print(f"report_sha256: {report.report_sha256}")
        for error in (*report.errors, *report.integrity_errors):
            print(f"error: {error}", file=sys.stderr)
    return 0 if report.status == "complete" else 2


def command_polymarket_resolve(args: argparse.Namespace) -> int:
    """Persist independently cross-checked official outcomes for paper evidence."""

    wait_seconds = int(args.wait_seconds)
    poll_seconds = int(args.poll_interval_seconds)
    if wait_seconds < 0 or wait_seconds > 3_600:
        print("polymarket-resolve failed: --wait-seconds must lie in [0, 3600]", file=sys.stderr)
        return 2
    if poll_seconds < 1 or poll_seconds > 300:
        print(
            "polymarket-resolve failed: --poll-interval-seconds must lie in [1, 300]",
            file=sys.stderr,
        )
        return 2
    try:
        with PolymarketEvidenceStore(
            Path(args.database),
            memory_limit=str(args.memory_limit),
            threads=int(args.database_threads),
        ) as store:
            selected = str(getattr(args, "run_id", None) or "").strip()
            if not selected:
                row = store.connect().execute(
                    """
                    SELECT run_id FROM polymarket_recorder_run
                    WHERE status IN ('complete', 'degraded')
                    ORDER BY ended_at_ms DESC, run_id DESC LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    raise ValueError("no finished Polymarket recorder run is available")
                selected = str(row[0])
            finalizer = PolymarketResolutionFinalizer(store)
            deadline = time.monotonic() + wait_seconds
            while True:
                report = finalizer.finalize(run_id=selected)
                remaining = deadline - time.monotonic()
                if report.status == "complete" or remaining <= 0:
                    break
                time.sleep(min(float(poll_seconds), remaining))
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(
            f"polymarket-resolve failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    payload = report.asdict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "polymarket-resolve: "
            f"status={report.status} run={report.run_id} "
            f"finalized={report.finalized_count}/{report.market_count} "
            f"new={report.newly_finalized_count}"
        )
        for condition_id in report.pending_condition_ids:
            print(f"pending: {condition_id}")
    return 0 if report.status == "complete" else 2


def command_polymarket_features(args: argparse.Namespace) -> int:
    """Materialize causal feature rows from one immutable prospective run."""

    try:
        with PolymarketEvidenceStore(
            Path(args.database),
            memory_limit=str(args.memory_limit),
            threads=int(args.database_threads),
        ) as store:
            dataset = build_polymarket_feature_dataset(
                store,
                run_id=getattr(args, "run_id", None),
                config=PolymarketFeatureConfig(
                    cadence_ms=int(args.cadence_ms),
                    warmup_ms=int(args.warmup_ms),
                    minimum_resolved_markets_per_asset=int(
                        args.minimum_resolved_markets_per_asset
                    ),
                    allow_segmented_gaps=bool(
                        getattr(args, "allow_segmented_gaps", False)
                    ),
                ),
            )
            materialization = materialize_polymarket_feature_dataset(store, dataset)
            payload = dataset.summary()
            payload["materialization"] = materialization.asdict()
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(
            f"polymarket-features failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "polymarket-features: "
            f"run={dataset.run_id} rows={len(dataset.rows)} "
            f"labeled_rows={dataset.labeled_row_count} "
            f"shadow_ready={dataset.shadow_ready} "
            f"training_ready={dataset.training_ready} "
            f"materialization={materialization.status}"
        )
        print(f"dataset_sha256: {dataset.dataset_sha256}")
    return 0 if dataset.shadow_ready else 2


def command_polymarket_action_value(args: argparse.Namespace) -> int:
    """Materialize the frozen Round 9 labels in resumable condition batches."""

    started = time.monotonic()

    def progress(phase: str, payload: Mapping[str, object]) -> None:
        if getattr(args, "json", False):
            return
        details = " ".join(
            f"{key}={value}" for key, value in sorted(payload.items())
        )
        print(
            "polymarket-action-value-progress: "
            f"phase={phase} elapsed_seconds={time.monotonic() - started:.1f}"
            + (f" {details}" if details else ""),
            file=sys.stderr,
            flush=True,
        )

    try:
        with PolymarketEvidenceStore(
            Path(args.database),
            memory_limit=str(args.memory_limit),
            threads=int(args.database_threads),
        ) as store:
            pipeline_config = PolymarketActionPipelineConfig(
                market_groups_per_batch=int(args.market_groups_per_batch),
            )
            eligible_condition_ids = None
            eligibility_sha256 = ""
            if bool(getattr(args, "allow_segmented_gaps", False)):
                continuity = evaluate_polymarket_continuity_eligibility(
                    store,
                    run_id=str(args.run_id),
                )
                if not continuity.confirmation_eligible:
                    raise ValueError(
                        "segmented continuity is insufficient: "
                        + "; ".join(continuity.confirmation_reasons)
                    )
                eligible_condition_ids = continuity.eligible_condition_ids
                eligibility_sha256 = continuity.report_sha256
                pipeline_config = replace(
                    pipeline_config,
                    feature=replace(
                        pipeline_config.feature,
                        allow_segmented_gaps=True,
                    ),
                )
            report = materialize_polymarket_action_value_batches(
                store,
                run_id=str(args.run_id),
                config=pipeline_config,
                eligible_condition_ids=eligible_condition_ids,
                eligibility_sha256=eligibility_sha256,
                progress=progress,
            )
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(
            f"polymarket-action-value failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    payload = report.asdict()
    payload["batch_materialization_status"] = [
        item.status for item in report.batches
    ]
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        created = sum(item.status == "created" for item in report.batches)
        print(
            "polymarket-action-value: "
            f"run={report.run_id} batches={len(report.batches)} "
            f"created={created} actions={report.action_count} "
            f"classifier_eligible={report.classifier_eligible_count} "
            f"positive_complete={report.positive_complete_count}"
        )
        print(f"report_sha256: {report.report_sha256}")
    return 0


def command_polymarket_continuity(args: argparse.Namespace) -> int:
    """Audit synchronized evidence windows without target-derived information."""

    try:
        with PolymarketEvidenceStore(
            Path(args.database),
            memory_limit=str(args.memory_limit),
            threads=int(args.database_threads),
        ) as store:
            report = evaluate_polymarket_continuity_eligibility(
                store,
                run_id=str(args.run_id),
            )
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(
            f"polymarket-continuity failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    payload = report.asdict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "polymarket-continuity: "
            f"run={report.run_id} eligible_groups={report.eligible_group_count}/"
            f"{len(report.groups)} confirmation_eligible="
            f"{report.confirmation_eligible}"
        )
        for reason in report.confirmation_reasons:
            print(f"reason: {reason}")
        print(f"report_sha256: {report.report_sha256}")
    return 0 if report.confirmation_eligible else 2


def command_polymarket_ridge(args: argparse.Namespace) -> int:
    """Fit and persist the frozen causal ridge baseline from approved evidence."""

    try:
        with PolymarketEvidenceStore(
            Path(args.database),
            memory_limit=str(args.memory_limit),
            threads=int(args.database_threads),
        ) as store:
            dataset = load_polymarket_ridge_dataset(
                store,
                pipeline_report_sha256=str(args.pipeline_report_sha256),
            )
            report = fit_and_evaluate_polymarket_ridge(dataset)
            materialization = materialize_polymarket_ridge_report(
                store,
                dataset,
                report,
            )
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(
            f"polymarket-ridge failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    payload = report.asdict()
    payload["materialization"] = materialization.asdict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "polymarket-ridge: "
            f"policy={report.selected_policy} "
            f"threshold={report.selected_threshold} "
            f"test_completed={report.test_metrics.completed_trade_count} "
            f"test_stress_utility_quote="
            f"{report.test_metrics.aggregate_stress_utility_quote:.8f} "
            f"development_passed={report.development_passed} "
            f"materialization={materialization.status}"
        )
        for reason in report.test_metrics.gate_reasons:
            print(f"reason: {reason}")
        print(f"report_sha256: {report.report_sha256}")
    return 0 if report.development_passed else 2


def _polymarket_execution_uplift_metrics(
    report: object,
    *,
    dataset_fingerprint: str,
) -> dict[str, object]:
    trades = [item for item in getattr(report, "trades") if item.filled]
    values = [float(item.realized_pnl_quote) for item in trades]
    gains = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value < 0.0))
    loss_streak = 0
    maximum_loss_streak = 0
    for value in values:
        loss_streak = loss_streak + 1 if value < 0.0 else 0
        maximum_loss_streak = max(maximum_loss_streak, loss_streak)
    drawdown = float(getattr(report, "maximum_drawdown_fraction"))
    net = float(getattr(report, "net_realized_pnl_quote"))
    return {
        "realized_pnl": net,
        "roi_pct": 100.0 * float(getattr(report, "return_on_initial_capital")),
        "max_drawdown": drawdown,
        "expectancy": net / len(values) if values else 0.0,
        "profit_factor": gains / losses if losses > 0.0 else (gains if gains > 0.0 else 0.0),
        "closed_trades": len(values),
        "win_rate": (
            sum(value > 0.0 for value in values) / len(values) if values else 0.0
        ),
        "liquidation_events": 0,
        "max_consecutive_losses": maximum_loss_streak,
        "downside_return_risk_ratio": net / drawdown if drawdown > 0.0 else 0.0,
        "dataset_fingerprint": dataset_fingerprint,
        "evidence_sha256": str(getattr(report, "report_sha256")),
    }


def _polymarket_matched_uplift_periods(
    split: object,
    baseline: object,
    ai: object,
) -> list[dict[str, object]]:
    baseline_by_start = {
        item.event_start_ms: float(item.group_realized_pnl_quote)
        for item in getattr(baseline, "equity_curve")
    }
    ai_by_start = {
        item.event_start_ms: float(item.group_realized_pnl_quote)
        for item in getattr(ai, "equity_curve")
    }
    return [
        {
            "scope": "polymarket_btc_eth_sol_five_minute_test",
            "period_start_ms": int(start_ms),
            "period_end_ms": int(start_ms) + 300_000,
            "baseline_return": baseline_by_start.get(int(start_ms), 0.0),
            "ai_return": ai_by_start.get(int(start_ms), 0.0),
        }
        for start_ms in getattr(split, "test_group_starts_ms")
    ]


def _polymarket_held_out_prediction_evidence(
    samples: Sequence[Any],
    baseline_probabilities: Iterable[float],
    model_probabilities: Iterable[float],
) -> dict[str, object]:
    sample_rows = tuple(samples)
    baseline_rows = tuple(float(value) for value in baseline_probabilities)
    model_rows = tuple(float(value) for value in model_probabilities)
    if not sample_rows or not (
        len(sample_rows) == len(baseline_rows) == len(model_rows)
    ):
        raise ValueError("held-out Polymarket prediction evidence is incomplete")
    if any(
        format(item.baseline_up_probability, ".17g") != format(baseline, ".17g")
        for item, baseline in zip(sample_rows, baseline_rows, strict=True)
    ):
        raise ValueError("held-out market priors do not match model samples")
    rows = [
        {
            **item.asdict(),
            "model_up_probability": format(model, ".17g"),
        }
        for item, _baseline, model in zip(
            sample_rows,
            baseline_rows,
            model_rows,
            strict=True,
        )
    ]
    canonical = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        "schema_version": "polymarket-held-out-predictions-v2",
        "role": "untouched_chronological_test",
        "row_count": len(rows),
        "market_count": len({item.condition_id for item in sample_rows}),
        "time_group_count": len({item.event_start_ms for item in sample_rows}),
        "assets": sorted({item.asset for item in sample_rows}),
        "rows_sha256": hashlib.sha256(canonical).hexdigest(),
        "rows": rows,
    }


def _polymarket_profile_prediction_evidence(
    samples: Sequence[Any],
    profile_probabilities: Iterable[float],
    *,
    control_rows_sha256: str,
) -> dict[str, object]:
    sample_rows = tuple(samples)
    probabilities = tuple(float(value) for value in profile_probabilities)
    if (
        not sample_rows
        or len(sample_rows) != len(probabilities)
        or len(control_rows_sha256) != 64
        or any(not 0.0 < value < 1.0 for value in probabilities)
    ):
        raise ValueError("held-out Polymarket profile evidence is incomplete")
    rows = [
        {
            "sample_id": sample.sample_id,
            "profile_model_up_probability": format(probability, ".17g"),
        }
        for sample, probability in zip(
            sample_rows,
            probabilities,
            strict=True,
        )
    ]
    canonical = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        "schema_version": "polymarket-profile-held-out-predictions-v1",
        "role": "untouched_chronological_test",
        "control_rows_sha256": control_rows_sha256,
        "row_count": len(rows),
        "rows_sha256": hashlib.sha256(canonical).hexdigest(),
        "rows": rows,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
    }


def _polymarket_latency_scenarios(
    value: object,
    *,
    primary_latency_ms: int,
) -> tuple[int, ...]:
    raw = [item.strip() for item in str(value or "").split(",") if item.strip()]
    try:
        latencies = {int(item) for item in raw}
    except ValueError as exc:
        raise ValueError("Polymarket latency stress values must be integers") from exc
    latencies.add(int(primary_latency_ms))
    if (
        not latencies
        or len(latencies) > 12
        or any(not 1 <= latency <= 60_000 for latency in latencies)
    ):
        raise ValueError(
            "Polymarket latency stress values must contain 1-12 values in [1, 60000]"
        )
    return tuple(sorted(latencies))


def command_polymarket_model(args: argparse.Namespace) -> int:
    """Run one hash-bound probability and full-depth execution experiment."""

    started_monotonic = time.monotonic()

    def progress(phase: str, **details: object) -> None:
        if getattr(args, "json", False):
            return
        suffix = " ".join(
            f"{key}={value}" for key, value in sorted(details.items())
        )
        print(
            "polymarket-model-progress: "
            f"phase={phase} elapsed_seconds={time.monotonic() - started_monotonic:.1f}"
            + (f" {suffix}" if suffix else ""),
            file=sys.stderr,
            flush=True,
        )

    try:
        minimum_markets = int(args.minimum_resolved_markets_per_asset)
        latency_scenarios = _polymarket_latency_scenarios(
            args.latency_stress_ms,
            primary_latency_ms=int(args.latency_ms),
        )
        progress(
            "feature-build",
            minimum_markets_per_asset=minimum_markets,
            run_id=getattr(args, "run_id", None) or "latest",
        )
        with PolymarketEvidenceStore(
            Path(args.database),
            memory_limit=str(args.memory_limit),
            threads=int(args.database_threads),
        ) as store:
            allow_segmented_gaps = bool(
                getattr(args, "allow_segmented_gaps", False)
            )
            feature_dataset = build_polymarket_feature_dataset(
                store,
                run_id=getattr(args, "run_id", None),
                config=PolymarketFeatureConfig(
                    cadence_ms=int(args.cadence_ms),
                    warmup_ms=int(args.warmup_ms),
                    minimum_resolved_markets_per_asset=minimum_markets,
                    allow_segmented_gaps=allow_segmented_gaps,
                ),
            )
            progress(
                "feature-materialization",
                candidate_count=feature_dataset.candidate_count,
                row_count=len(feature_dataset.rows),
            )
            feature_materialization = materialize_polymarket_feature_dataset(
                store,
                feature_dataset,
            )
            progress("market-load")
            markets = PolymarketEvidenceReplay.load_markets(
                store,
                run_id=feature_dataset.run_id,
            )
            progress("model-dataset", market_count=len(markets))
            model_dataset = build_polymarket_model_dataset(
                feature_dataset,
                markets,
                config=PolymarketModelConfig(
                    minimum_markets_per_asset=minimum_markets,
                    minimum_time_groups=minimum_markets,
                ),
            )
            progress(
                "chronological-split",
                sample_count=len(model_dataset.samples),
                time_group_count=model_dataset.time_group_count,
            )
            split = split_polymarket_model_dataset(model_dataset)
            progress(
                "model-fit",
                test_groups=len(split.test_group_starts_ms),
                train_groups=len(split.train_group_starts_ms),
                validation_groups=len(split.validation_group_starts_ms),
            )
            model, probability_report = fit_polymarket_offset_model(
                model_dataset,
                split,
            )
            progress("profile-challenger-fit")
            profile_model, profile_probability_report = (
                fit_polymarket_profile_challenger(
                    model_dataset,
                    split,
                    model,
                )
            )
            test_conditions = tuple(
                sorted({item.condition_id for item in split.test})
            )
            progress(
                "execution-replay-load",
                test_condition_count=len(test_conditions),
            )
            execution_replay = PolymarketEvidenceReplay.load(
                store,
                run_id=feature_dataset.run_id,
                allow_segmented_gaps=allow_segmented_gaps,
                book_sample_interval_ms=0,
                condition_ids=test_conditions,
            )
            baseline_probabilities = [
                item.baseline_up_probability for item in split.test
            ]
            model_probabilities = predict_polymarket_probabilities(
                model,
                split.test,
            )
            profile_probabilities = predict_polymarket_profile_probabilities(
                profile_model,
                split.test,
            )
            progress(
                "held-out-evidence",
                selected_candidate=model.selected_candidate,
                test_sample_count=len(split.test),
            )
            prediction_evidence = _polymarket_held_out_prediction_evidence(
                split.test,
                baseline_probabilities,
                model_probabilities,
            )
            profile_prediction_evidence = _polymarket_profile_prediction_evidence(
                split.test,
                profile_probabilities,
                control_rows_sha256=str(prediction_evidence["rows_sha256"]),
            )
            execution_config = PolymarketExecutionResearchConfig(
                submission_latency_ms=int(args.latency_ms),
                maximum_execution_observation_delay_ms=int(
                    args.max_execution_observation_delay_ms
                ),
                minimum_expected_edge_per_contract=str(args.minimum_edge),
                initial_capital_quote=str(args.initial_capital),
                maximum_loss_fraction_per_market=str(
                    args.maximum_loss_fraction_per_market
                ),
                maximum_loss_fraction_per_time_group=str(
                    args.maximum_loss_fraction_per_time_group
                ),
            ).validated()
            progress("execution-baseline", latency_ms=execution_config.submission_latency_ms)
            baseline_execution = evaluate_polymarket_execution_policy(
                split.test,
                baseline_probabilities,
                execution_replay,
                config=execution_config,
            )
            progress("execution-model", latency_ms=execution_config.submission_latency_ms)
            model_execution = evaluate_polymarket_execution_policy(
                split.test,
                model_probabilities,
                execution_replay,
                config=execution_config,
            )
            progress(
                "execution-profile-model",
                latency_ms=execution_config.submission_latency_ms,
            )
            profile_model_execution = evaluate_polymarket_execution_policy(
                split.test,
                profile_probabilities,
                execution_replay,
                config=execution_config,
            )
            progress(
                "execution-model-retry",
                latency_ms=execution_config.submission_latency_ms,
            )
            model_retry_execution = evaluate_polymarket_retry_execution_policy(
                split.test,
                model_probabilities,
                execution_replay,
                config=execution_config,
            )
            latency_sensitivity: dict[str, dict[int, Any]] = {
                "baseline": {},
                "model": {},
                "profile_model": {},
                "model_retry": {},
            }
            for latency_ms in latency_scenarios:
                progress("latency-sensitivity", latency_ms=latency_ms)
                scenario_config = replace(
                    execution_config,
                    submission_latency_ms=latency_ms,
                ).validated()
                latency_sensitivity["baseline"][latency_ms] = (
                    baseline_execution
                    if latency_ms == execution_config.submission_latency_ms
                    else evaluate_polymarket_execution_policy(
                        split.test,
                        baseline_probabilities,
                        execution_replay,
                        config=scenario_config,
                    )
                )
                latency_sensitivity["model"][latency_ms] = (
                    model_execution
                    if latency_ms == execution_config.submission_latency_ms
                    else evaluate_polymarket_execution_policy(
                        split.test,
                        model_probabilities,
                        execution_replay,
                        config=scenario_config,
                    )
                )
                latency_sensitivity["profile_model"][latency_ms] = (
                    profile_model_execution
                    if latency_ms == execution_config.submission_latency_ms
                    else evaluate_polymarket_execution_policy(
                        split.test,
                        profile_probabilities,
                        execution_replay,
                        config=scenario_config,
                    )
                )
                latency_sensitivity["model_retry"][latency_ms] = (
                    model_retry_execution
                    if latency_ms == execution_config.submission_latency_ms
                    else evaluate_polymarket_retry_execution_policy(
                        split.test,
                        model_probabilities,
                        execution_replay,
                        config=scenario_config,
                    )
                )
        ai_payload: dict[str, object]
        ai_execution = None
        ai_uplift = None
        if bool(getattr(args, "disable_ai", False)):
            ai_payload = {
                "enabled": False,
                "reason": "operator_disabled",
                "trading_authority": False,
                "profitability_claim": False,
            }
        else:
            from .ai_model_benchmark import (
                AI_MODEL_BENCHMARK_CONTRACT,
                rescore_finance_ai_benchmark_payload,
            )
            from .ai_uplift import assess_ai_uplift

            progress("ai-benchmark-verification", model=str(args.ai_model))
            benchmark_path = Path(str(args.ai_benchmark))
            benchmark_bytes = benchmark_path.read_bytes()
            benchmark_payload = json.loads(benchmark_bytes.decode("utf-8"))
            rescored_benchmark = rescore_finance_ai_benchmark_payload(
                benchmark_payload
            )
            ai_model = str(args.ai_model)
            selected_result = next(
                (
                    item
                    for item in rescored_benchmark.results
                    if item.model == ai_model and item.passed
                ),
                None,
            )
            if (
                rescored_benchmark.benchmark_contract
                != AI_MODEL_BENCHMARK_CONTRACT
                or rescored_benchmark.selected_model != ai_model
                or selected_result is None
            ):
                raise ValueError(
                    "the frozen adversarial risk benchmark does not select "
                    f"the requested AI model: {ai_model}"
                )
            benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
            policy_selection = build_polymarket_policy_selection(
                split.test,
                model_probabilities,
                execution_replay.markets,
                config=execution_config,
            )
            ai_cases = build_polymarket_ai_veto_cases(
                policy_selection,
                probability_report,
                execution_config,
            )
            progress("ai-veto", case_count=len(ai_cases), model=ai_model)

            def ai_progress(_event: str, item: Mapping[str, object]) -> None:
                if not getattr(args, "json", False):
                    print(
                        "polymarket-ai: "
                        f"case={item.get('case')}/{item.get('case_count')} "
                        f"action={item.get('action')} "
                        f"valid={item.get('valid')} "
                        f"latency={item.get('latency_seconds')}s",
                        file=sys.stderr,
                    )

            ai_report = benchmark_polymarket_ai_veto(
                ai_cases,
                all_condition_ids=[item.condition_id for item in split.test],
                selection_sha256=policy_selection.selection_sha256,
                risk_benchmark_evidence_sha256=benchmark_sha256,
                config=PolymarketAIVetoConfig(
                    model=ai_model,
                    base_url=str(args.ai_url),
                    timeout_seconds=float(args.ai_timeout),
                    minimum_approval_confidence=float(args.ai_min_confidence),
                    maximum_advisory_latency_seconds=float(
                        args.ai_max_latency_seconds
                    ),
                ),
                progress=ai_progress,
            )
            ai_decision_delays = {
                condition_id: 0
                for condition_id in {item.condition_id for item in split.test}
            }
            for result in ai_report.results:
                ai_decision_delays[result.condition_id] = int(
                    math.ceil(max(0.0, result.latency_seconds) * 1_000.0)
                )
            ai_execution = evaluate_polymarket_execution_policy(
                split.test,
                model_probabilities,
                execution_replay,
                config=execution_config,
                market_permissions=ai_report.market_permissions,
                decision_delay_ms_by_condition=ai_decision_delays,
            )
            latency_sensitivity["ai"] = {}
            for latency_ms in latency_scenarios:
                scenario_config = replace(
                    execution_config,
                    submission_latency_ms=latency_ms,
                ).validated()
                latency_sensitivity["ai"][latency_ms] = (
                    ai_execution
                    if latency_ms == execution_config.submission_latency_ms
                    else evaluate_polymarket_execution_policy(
                        split.test,
                        model_probabilities,
                        execution_replay,
                        config=scenario_config,
                        market_permissions=ai_report.market_permissions,
                        decision_delay_ms_by_condition=ai_decision_delays,
                    )
                )
            ai_uplift = assess_ai_uplift(
                _polymarket_execution_uplift_metrics(
                    model_execution,
                    dataset_fingerprint=model_dataset.dataset_sha256,
                ),
                _polymarket_execution_uplift_metrics(
                    ai_execution,
                    dataset_fingerprint=model_dataset.dataset_sha256,
                ),
                model_name=ai_model,
                model_parameters_b=ai_report.model_parameters_b,
                model_artifact_sha256=ai_report.report_sha256,
                matched_periods=_polymarket_matched_uplift_periods(
                    split,
                    model_execution,
                    ai_execution,
                ),
            )
            progress(
                "ai-uplift",
                accepted=ai_uplift.accepted,
                filled_orders=ai_execution.filled_order_count,
            )
            ai_payload = {
                "enabled": True,
                "risk_benchmark": {
                    "path": str(benchmark_path),
                    "sha256": benchmark_sha256,
                    "contract": rescored_benchmark.benchmark_contract,
                    "selected_model": rescored_benchmark.selected_model,
                    "score": selected_result.score,
                },
                "policy_selection": policy_selection.asdict(),
                "prompt_cases": [case.asdict() for case in ai_cases],
                "veto_report": ai_report.asdict(),
                "execution": ai_execution.asdict(),
                "uplift": ai_uplift.asdict(),
            }
        execution_reports = [
            report
            for policy_reports in latency_sensitivity.values()
            for report in policy_reports.values()
        ]
        probability_gates_passed = (
            probability_report.validation_log_loss_delta < 0.0
            and probability_report.test_log_loss_delta < 0.0
            and probability_report.test_brier_delta < 0.0
        )
        retry_latency_reports = latency_sensitivity["model_retry"]
        retry_control_reports = latency_sensitivity["model"]
        retry_gates = {
            "probability_model_gates_passed": probability_gates_passed,
            "minimum_confirmatory_test_time_groups_met": (
                len(split.test_group_starts_ms) >= 30
            ),
            "positive_after_cost_at_every_latency": all(
                retry_latency_reports[latency].net_realized_pnl_quote > 0
                for latency in latency_scenarios
            ),
            "improved_after_cost_at_every_latency": all(
                retry_latency_reports[latency].net_realized_pnl_quote
                > retry_control_reports[latency].net_realized_pnl_quote
                for latency in latency_scenarios
            ),
            "return_on_deployed_not_worse_at_every_latency": all(
                retry_latency_reports[latency].return_on_deployed_capital
                >= retry_control_reports[latency].return_on_deployed_capital
                for latency in latency_scenarios
            ),
            "maximum_drawdown_not_worse_at_every_latency": all(
                retry_latency_reports[latency].maximum_drawdown_fraction
                <= retry_control_reports[latency].maximum_drawdown_fraction
                for latency in latency_scenarios
            ),
            "all_order_outcomes_terminal": all(
                trade.execution_state != "UNKNOWN"
                for report in retry_latency_reports.values()
                for trade in report.trades
            ),
        }
        retry_accepted = all(retry_gates.values())
        retry_challenger = {
            "schema_version": POLYMARKET_RETRY_CHALLENGER_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_RETRY_CONTRACT_SHA256,
            "control_policy": "model",
            "challenger_policy": "model_retry",
            "gates": retry_gates,
            "accepted": retry_accepted,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        profile_latency_reports = latency_sensitivity["profile_model"]
        profile_control_reports = latency_sensitivity["model"]
        profile_gates = {
            "validation_log_loss_not_worse_than_control": (
                profile_probability_report.validation_log_loss_delta_vs_control
                <= 0.0
            ),
            "test_log_loss_strictly_better_than_control": (
                profile_probability_report.test_log_loss_delta_vs_control < 0.0
            ),
            "test_brier_score_strictly_better_than_control": (
                profile_probability_report.test_brier_delta_vs_control < 0.0
            ),
            "minimum_untouched_test_time_groups_met": (
                len(split.test_group_starts_ms) >= 30
            ),
            "positive_after_cost_at_every_latency": all(
                profile_latency_reports[latency].net_realized_pnl_quote > 0
                for latency in latency_scenarios
            ),
            "improved_after_cost_at_every_latency": all(
                profile_latency_reports[latency].net_realized_pnl_quote
                > profile_control_reports[latency].net_realized_pnl_quote
                for latency in latency_scenarios
            ),
            "return_on_deployed_not_worse_at_every_latency": all(
                profile_latency_reports[latency].return_on_deployed_capital
                >= profile_control_reports[latency].return_on_deployed_capital
                for latency in latency_scenarios
            ),
            "maximum_drawdown_not_worse_at_every_latency": all(
                profile_latency_reports[latency].maximum_drawdown_fraction
                <= profile_control_reports[latency].maximum_drawdown_fraction
                for latency in latency_scenarios
            ),
            "all_order_outcomes_terminal": all(
                trade.execution_state != "UNKNOWN"
                for report in profile_latency_reports.values()
                for trade in report.trades
            ),
        }
        profile_promotion_gates_passed = all(profile_gates.values())
        profile_challenger = {
            "schema_version": POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_PROFILE_CONTRACT_SHA256,
            "control_policy": "model",
            "challenger_policy": "profile_model",
            "gates": profile_gates,
            "promotion_gates_passed": profile_promotion_gates_passed,
            "accepted": False,
            "status": (
                "awaiting_later_prospective_confirmation"
                if profile_promotion_gates_passed
                else "exploratory_gates_failed"
            ),
            "requires_later_prospective_confirmation": True,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        latency_sensitivity_payload = {
            "schema_version": "polymarket-execution-latency-sensitivity-v1",
            "primary_network_latency_ms": execution_config.submission_latency_ms,
            "network_latencies_ms": list(latency_scenarios),
            "policies": {
                policy: {
                    str(latency): reports[latency].asdict()
                    for latency in latency_scenarios
                }
                for policy, reports in latency_sensitivity.items()
            },
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        evidence_gates = {
            "validation_probability_improved": (
                probability_report.validation_log_loss_delta < 0.0
            ),
            "untouched_test_probability_improved": (
                probability_report.test_log_loss_delta < 0.0
                and probability_report.test_brier_delta < 0.0
            ),
            "minimum_confirmatory_test_time_groups_met": (
                len(split.test_group_starts_ms) >= 30
            ),
            "after_cost_execution_improved": (
                model_execution.net_realized_pnl_quote
                > baseline_execution.net_realized_pnl_quote
            ),
            "after_cost_model_improved_at_every_stress_latency": all(
                latency_sensitivity["model"][latency].net_realized_pnl_quote
                > latency_sensitivity["baseline"][latency].net_realized_pnl_quote
                for latency in latency_scenarios
            ),
            "retry_challenger_accepted": retry_accepted,
            "profile_challenger_promotion_gates_passed": (
                profile_promotion_gates_passed
            ),
            "all_positions_officially_settled": (
                all(
                    report.filled_order_count
                    == report.winning_order_count + report.losing_order_count
                    and all(
                        not trade.filled or trade.official_resolution_event_id
                        for trade in report.trades
                    )
                    for report in execution_reports
                )
            ),
            "all_order_outcomes_terminal": (
                all(
                    trade.execution_state != "UNKNOWN"
                    for report in execution_reports
                    for trade in report.trades
                )
            ),
            "ai_enabled": bool(ai_payload.get("enabled")),
            "ai_uplift_accepted": bool(
                ai_uplift is not None and ai_uplift.accepted
            ),
            "live_trading_authority": False,
            "profitability_claim": False,
        }
        payload: dict[str, object] = {
            "schema_version": POLYMARKET_MODEL_ARTIFACT_SCHEMA_VERSION,
            "run_id": feature_dataset.run_id,
            "feature_dataset": feature_dataset.summary(),
            "feature_materialization": feature_materialization.asdict(),
            "model_dataset": model_dataset.summary(),
            "split": split.summary(),
            "model": model.asdict(),
            "probability_report": probability_report.asdict(),
            "profile_model": profile_model.asdict(),
            "profile_probability_report": profile_probability_report.asdict(),
            "held_out_prediction_evidence": prediction_evidence,
            "profile_held_out_prediction_evidence": (
                profile_prediction_evidence
            ),
            "baseline_execution": baseline_execution.asdict(),
            "model_execution": model_execution.asdict(),
            "profile_model_execution": profile_model_execution.asdict(),
            "model_retry_execution": model_retry_execution.asdict(),
            "retry_challenger": retry_challenger,
            "profile_challenger": profile_challenger,
            "execution_latency_sensitivity": latency_sensitivity_payload,
            "ai": ai_payload,
            "confirmatory_evidence_contract": {
                "independent_unit": "shared_btc_eth_sol_five_minute_time_group",
                "minimum_untouched_test_time_groups": 30,
                "observed_untouched_test_time_groups": len(
                    split.test_group_starts_ms
                ),
                "minimum_markets_per_asset": minimum_markets,
                "confirmatory_ready": len(split.test_group_starts_ms) >= 30,
                "trading_authority": False,
                "profitability_claim": False,
            },
            "evidence_gates": evidence_gates,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        artifact_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        payload["artifact_sha256"] = artifact_sha256
        output = str(getattr(args, "output", None) or "").strip()
        if output:
            write_json_atomic(Path(output), payload, sort_keys=True)
        progress("complete", artifact_sha256=artifact_sha256)
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(
            f"polymarket-model failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "polymarket-model: "
            f"run={feature_dataset.run_id} "
            f"candidate={model.selected_candidate} "
            f"test_log_loss_delta={probability_report.test_log_loss_delta:.8f} "
            f"baseline_net={baseline_execution.net_realized_pnl_quote} "
            f"model_net={model_execution.net_realized_pnl_quote} "
            f"model_fills={model_execution.filled_order_count} "
            f"profile_net={profile_model_execution.net_realized_pnl_quote} "
            f"profile_fills={profile_model_execution.filled_order_count} "
            f"profile_gates={profile_promotion_gates_passed} "
            f"retry_net={model_retry_execution.net_realized_pnl_quote} "
            f"retry_fills={model_retry_execution.filled_order_count} "
            f"retry_accepted={retry_accepted}"
        )
        if ai_execution is not None and ai_uplift is not None:
            print(
                "polymarket-ai: "
                f"net={ai_execution.net_realized_pnl_quote} "
                f"fills={ai_execution.filled_order_count} "
                f"uplift_accepted={ai_uplift.accepted}"
            )
        print(f"artifact_sha256: {artifact_sha256}")
        if output:
            print(f"artifact: {output}")
    return 0


def command_polymarket_verify(args: argparse.Namespace) -> int:
    """Independently reconstruct a model artifact from its recorder database."""

    started = time.monotonic()

    def progress(phase: str, details: Mapping[str, object]) -> None:
        if getattr(args, "json", False):
            return
        suffix = " ".join(
            f"{key}={value}" for key, value in sorted(details.items())
        )
        print(
            "polymarket-verify-progress: "
            f"phase={phase} elapsed_seconds={time.monotonic() - started:.1f}"
            + (f" {suffix}" if suffix else ""),
            file=sys.stderr,
            flush=True,
        )

    try:
        report = verify_polymarket_model_artifact_source(
            args.artifact,
            args.database,
            memory_limit=str(args.memory_limit),
            database_threads=int(args.database_threads),
            progress=progress,
        )
        output = str(getattr(args, "output", None) or "").strip()
        if output:
            write_json_atomic(Path(output), report.asdict(), sort_keys=True)
    except Exception as exc:  # Verification is an explicit fail-closed boundary.
        print(
            f"polymarket-verify failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    payload = report.asdict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "polymarket-verify: "
            f"status={report.status} run={report.run_id} "
            f"scenarios={report.verified_execution_scenario_count} "
            f"trades={report.verified_execution_trade_count} "
            f"report_sha256={report.report_sha256}"
        )
        if output:
            print(f"verification: {output}")
    return 0


def command_polymarket_publish(args: argparse.Namespace) -> int:
    """Source-verify an artifact, then derive its current documentation."""

    started = time.monotonic()

    def progress(phase: str, details: Mapping[str, object]) -> None:
        if getattr(args, "json", False):
            return
        suffix = " ".join(
            f"{key}={value}" for key, value in sorted(details.items())
        )
        print(
            "polymarket-publish-progress: "
            f"phase={phase} elapsed_seconds={time.monotonic() - started:.1f}"
            + (f" {suffix}" if suffix else ""),
            file=sys.stderr,
            flush=True,
        )

    try:
        verification = verify_polymarket_model_artifact_source(
            args.artifact,
            args.database,
            memory_limit=str(args.memory_limit),
            database_threads=int(args.database_threads),
            progress=progress,
        )
        progress("documentation-publication", {"round": int(args.round)})
        result = publish_polymarket_model_artifact(
            args.artifact,
            args.research_root,
            round_number=int(args.round),
            prior_round_path=(
                str(args.prior_round).strip()
                if str(getattr(args, "prior_round", "") or "").strip()
                else None
            ),
            source_verification=verification.asdict(),
        )
    except Exception as exc:  # Publication is an explicit fail-closed boundary.
        print(
            f"polymarket-publish failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    payload = result.asdict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "polymarket-publish: "
            f"round={result.round_number} files={len(result.generated_files)} "
            f"artifact_sha256={result.artifact_sha256} "
            f"source_verification_sha256={verification.report_sha256} "
            f"manifest_sha256={result.manifest_sha256}"
        )
        print(f"research_root: {result.research_root}")
    return 0


def command_polymarket_paper(args: argparse.Namespace) -> int:
    """Inspect or execute evidence-bound Polymarket paper lifecycle actions."""

    action = str(getattr(args, "action", "status") or "status")
    stop_succeeded: bool | None = None
    model_run_succeeded: bool | None = None

    def required(name: str) -> str:
        value = str(getattr(args, name, None) or "").strip()
        if not value:
            raise ValueError(f"--{name.replace('_', '-')} is required for {action}")
        return value

    try:
        model_plan = None
        broker_run_id = getattr(args, "run_id", None)
        maximum_observation_delay_ms = int(
            args.max_execution_observation_delay_ms
        )
        maximum_book_age_ms = 2_000
        order_ttl_ms = 30_000
        allow_segmented_gaps = bool(
            getattr(args, "allow_segmented_gaps", False)
        )
        if action == "run-model":
            artifact_path = Path(required("artifact"))
            source_verification_path = Path(required("source_verification"))
            model_plan = build_polymarket_paper_plan(
                artifact_path,
                source_verification_path,
                policy=str(getattr(args, "policy", "auto")),
                allow_unconfirmed_research=bool(
                    getattr(args, "allow_unconfirmed_research", False)
                ),
            )
            if allow_segmented_gaps and not model_plan.allow_segmented_gaps:
                raise ValueError(
                    "--allow-segmented-gaps disagrees with the verified model artifact"
                )
            allow_segmented_gaps = model_plan.allow_segmented_gaps
            supplied_run_id = str(broker_run_id or "").strip()
            if supplied_run_id and supplied_run_id != model_plan.run_id:
                raise ValueError("--run-id disagrees with the verified model artifact")
            broker_run_id = model_plan.run_id
            maximum_observation_delay_ms = int(
                model_plan.execution_config[
                    "maximum_execution_observation_delay_ms"
                ]
            )
            maximum_book_age_ms = int(
                model_plan.execution_config["maximum_book_age_ms"]
            )
            order_ttl_ms = int(model_plan.execution_config["order_ttl_ms"])
            output_value = str(getattr(args, "output", None) or "").strip()
            if output_value:
                output_path = Path(output_value).resolve()
                protected_paths = {
                    Path(args.database).resolve(),
                    artifact_path.resolve(),
                    source_verification_path.resolve(),
                }
                if output_path in protected_paths:
                    raise ValueError(
                        "--output must not overwrite the database or source evidence"
                    )
        elif any(
            (
                getattr(args, "artifact", None),
                getattr(args, "source_verification", None),
                getattr(args, "allow_unconfirmed_research", False),
                getattr(args, "output", None),
            )
        ):
            raise ValueError(
                "--artifact, --source-verification, --allow-unconfirmed-research, "
                "and --output require --action run-model"
            )

        with PolymarketPaperBroker(
            Path(args.database),
            run_id=broker_run_id,
            maximum_execution_observation_delay_ms=maximum_observation_delay_ms,
            maximum_book_age_ms=maximum_book_age_ms,
            order_ttl_ms=order_ttl_ms,
            allow_segmented_gaps=allow_segmented_gaps,
            memory_limit=str(args.memory_limit),
            threads=int(args.database_threads),
        ) as broker:
            coordinator = PolymarketPaperCoordinator(
                broker,
                control_path=getattr(args, "control_path", None),
            )
            operation: dict[str, object] = {"action": action}
            if action == "run-model":
                if model_plan is None:
                    raise RuntimeError("Polymarket paper model plan is unavailable")
                model_run = run_polymarket_paper_plan(
                    broker,
                    coordinator,
                    model_plan,
                )
                plan_summary = model_plan.asdict()
                plan_summary.pop("trades")
                operation["plan"] = plan_summary
                operation["model_run"] = model_run.asdict()
                model_run_succeeded = model_run.successful
                output_value = str(getattr(args, "output", None) or "").strip()
                if output_value:
                    write_json_atomic(
                        Path(output_value),
                        {
                            "plan": plan_summary,
                            "model_run": model_run.asdict(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
            elif action == "open":
                coordinator.require_open_allowed()
                event_id = required("event_id")
                outcome = required("outcome")
                matches = [
                    book
                    for book in broker.replay.books
                    if book.event_id == event_id and book.outcome == outcome
                ]
                if len(matches) != 1:
                    raise ValueError("--event-id and --outcome must select one replay book")
                position, result = broker.open_position(
                    position_id=required("position_id"),
                    decision=matches[0],
                    outcome=outcome,
                    quantity=required("quantity"),
                    maximum_price=required("limit_price"),
                    submission_latency_ms=int(required("latency_ms")),
                    decision_delay_ms=int(
                        getattr(args, "decision_delay_ms", 0)
                    ),
                    order_type=str(getattr(args, "order_type", "FAK")),
                )
                operation["position"] = None if position is None else asdict(position)
                operation["execution"] = asdict(result)
            elif action == "close":
                opening_id = required("opening_intent_id")
                position = next(
                    (
                        item
                        for item in broker.positions()
                        if item.opening_intent_id == opening_id
                    ),
                    None,
                )
                if position is None:
                    raise ValueError("--opening-intent-id has no open paper inventory")
                event_id = required("event_id")
                matches = [
                    book
                    for book in broker.replay.books
                    if book.event_id == event_id and book.token_id == position.token_id
                ]
                if len(matches) != 1:
                    raise ValueError("--event-id must select one replay book for the inventory")
                closed, result = broker.close_position(
                    opening_intent_id=opening_id,
                    decision=matches[0],
                    quantity=getattr(args, "quantity", None),
                    minimum_price=required("limit_price"),
                    submission_latency_ms=int(required("latency_ms")),
                )
                operation["close"] = None if closed is None else asdict(closed)
                operation["execution"] = asdict(result)
            elif action == "settle":
                opening_id = required("opening_intent_id")
                event_id = required("event_id")
                resolutions = [
                    item for item in broker.replay.resolutions if item.event_id == event_id
                ]
                if len(resolutions) != 1:
                    raise ValueError("--event-id must select one official resolution")
                operation["settlement"] = asdict(
                    broker.settle_position(
                        opening_intent_id=opening_id,
                        resolution=resolutions[0],
                    )
                )
            elif action == "pause":
                operation["control"] = coordinator.pause()
            elif action == "resume":
                operation["control"] = coordinator.resume()
            elif action == "stop":
                stop_report = coordinator.stop_all_positions(
                    submission_latency_ms=int(required("latency_ms")),
                )
                operation["stop"] = stop_report.asdict()
                stop_succeeded = stop_report.stopped
            elif action != "status":
                raise ValueError(f"unsupported Polymarket paper action: {action}")

            reconciliation = broker.reconcile()
            positions = (
                [asdict(position) for position in broker.positions()]
                if reconciliation.ok
                else []
            )
            payload = {
                "run_id": broker.replay.run_id,
                "control": coordinator.status(),
                "operation": operation,
                "reconciliation": reconciliation.asdict(),
                "replay_diagnostics": broker.replay.diagnostics.asdict(),
                "feed_coverage": inspect_polymarket_feed_coverage(
                    broker.store,
                    run_id=broker.replay.run_id,
                    allow_segmented_gaps=bool(
                        getattr(args, "allow_segmented_gaps", False)
                    ),
                ).asdict(),
                "positions": positions,
            }
    except Exception as exc:  # The CLI/UI boundary must return a stable failure code.
        print(
            f"polymarket-paper failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        report = payload["reconciliation"]
        model_run = payload["operation"].get("model_run")
        if isinstance(model_run, Mapping):
            print(
                "polymarket-paper: "
                f"action={action} run={payload['run_id']} "
                f"status={model_run['status']} policy={model_run['policy']} "
                f"matched={model_run['matched_execution_count']}/"
                f"{model_run['planned_trade_count']} "
                f"pnl={model_run['realized_pnl_quote']}"
            )
        else:
            print(
                "polymarket-paper: "
                f"action={action} run={payload['run_id']} "
                f"can_open={report['can_open']} can_close={report['can_close']} "
                f"positions={len(payload['positions'])}"
            )
    if model_run_succeeded is not None:
        return 0 if model_run_succeeded else 2
    if stop_succeeded is not None:
        return 0 if stop_succeeded else 2
    return 0 if reconciliation.can_open and reconciliation.can_close else 2


def command_archive_sync(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    interval = str(getattr(args, "interval", None) or runtime.interval)
    market_type = str(getattr(args, "market", "spot") or "spot")
    cadence = str(getattr(args, "cadence", "monthly") or "monthly")
    data_type_arg = getattr(args, "data_type", None)
    data_type = str(data_type_arg or ("aggTrades" if market_type == "futures" and interval == "1s" else "klines"))
    if data_type == "aggTrades" and interval != "1s":
        print("archive-sync aggTrades ingestion emits 1s candles; use --interval 1s", file=sys.stderr)
        return 2
    start_period = str(getattr(args, "start_period", "") or "").strip() or None
    end_period = str(getattr(args, "end_period", "") or "").strip() or None
    try:
        validate_archive_period_window(start_period=start_period, end_period=end_period)
    except ValueError as exc:
        print(f"archive-sync invalid period window: {exc}", file=sys.stderr)
        return 2
    plan_only = bool(getattr(args, "plan_only", False))
    quote_asset = str(getattr(args, "quote_asset", None) or runtime.quote_asset or "USDC").upper()
    quote_gate = quote_asset if getattr(args, "quote_asset", None) else None
    raw_symbols = str(getattr(args, "symbols", "") or "").strip()
    symbols = [item.strip().upper() for item in raw_symbols.split(",") if item.strip()]
    requested_top_symbols = 0
    prelisted_archive_urls: dict[str, list[str]] = {}
    history_rejections: list[dict[str, str]] = []
    if not symbols and int(getattr(args, "top_symbols", 0) or 0) > 0:
        top_symbols = max(1, int(getattr(args, "top_symbols", 0) or 0))
        requested_top_symbols = top_symbols
        max_scan = max(top_symbols, int(getattr(args, "max_scan", 250) or 250))
        try:
            ranking_client = BinanceClient(
                "",
                "",
                testnet=False,
                market_type=market_type,
                max_calls_per_minute=max(1, int(runtime.max_rate_calls_per_minute)),
            )
            selection = select_top_liquidity_symbols(
                ranking_client,
                load_strategy(),
                quote_asset=quote_asset,
                count=max_scan,
                max_scan=max_scan,
                strict_only=True,
            )
        except (BinanceAPIError, OSError, ValueError) as exc:
            print(f"archive-sync failed to rank high-liquidity symbols: {exc}", file=sys.stderr)
            return 2
        min_history_months = max(0, int(getattr(args, "min_history_months", 0) or 0))
        symbols = []
        history_rejections: list[dict[str, str]] = []
        for item in selection:
            if len(symbols) >= top_symbols:
                break
            if not is_supported_major_symbol(item.symbol, quote_asset):
                history_rejections.append({"symbol": item.symbol, "error": "unsupported_non_major_asset"})
                continue
            try:
                urls = list_archive_urls(
                    symbol=item.symbol,
                    interval=interval,
                    market_type=market_type,
                    cadence=cadence,
                    data_type=data_type,
                )
            except (OSError, ValueError) as exc:
                history_rejections.append({"symbol": item.symbol, "error": f"list_failed:{exc}"})
                continue
            if cadence == "monthly" and min_history_months > 0 and len(urls) < min_history_months:
                history_rejections.append({
                    "symbol": item.symbol,
                    "error": f"history_months_below_min:{len(urls)}/{min_history_months}",
                })
                continue
            symbols.append(item.symbol)
            prelisted_archive_urls[item.symbol] = urls
        if not symbols:
            print(f"archive-sync found no eligible high-liquidity {quote_asset} symbols", file=sys.stderr)
            for rejection in history_rejections:
                print(f"warning: {rejection['symbol']} {rejection['error']}", file=sys.stderr)
            return 2
    if not symbols:
        symbols = [str(getattr(args, "symbol", None) or runtime.symbol).upper()]
    invalid_symbols = [symbol for symbol in symbols if not is_supported_major_symbol(symbol, quote_gate)]
    if invalid_symbols:
        print(
            "archive-sync supports only BTC, ETH, and SOL symbols quoted in USDC or USDT: "
            + ",".join(invalid_symbols),
            file=sys.stderr,
        )
        return 2
    max_files = getattr(args, "max_files", None)
    max_files_int = None if max_files is None else max(0, int(max_files))
    all_results = []
    errors: list[dict[str, str]] = []
    archive_plans: list[dict[str, object]] = []
    selected_archive_urls: dict[str, list[str]] = {}
    for symbol in symbols:
        try:
            urls = prelisted_archive_urls.get(symbol)
            if urls is None:
                urls = list_archive_urls(
                    symbol=symbol,
                    interval=interval,
                    market_type=market_type,
                    cadence=cadence,
                    data_type=data_type,
                )
        except (OSError, ValueError) as exc:
            errors.append({"symbol": symbol, "error": f"list_failed:{exc}"})
            continue
        listed_count = len(urls)
        metadata_by_url = archive_listing_items_by_url(urls)
        listed_bytes = sum(int(item.size_bytes) for item in metadata_by_url.values())
        try:
            urls = filter_archive_urls_by_period(urls, start_period=start_period, end_period=end_period)
        except ValueError as exc:
            errors.append({"symbol": symbol, "error": f"period_filter_failed:{exc}"})
            continue
        filtered_count = len(urls)
        filtered_bytes = sum(int(metadata_by_url[url].size_bytes) for url in urls if url in metadata_by_url)
        if max_files_int is not None:
            urls = urls[:max_files_int]
        selected_bytes = sum(int(metadata_by_url[url].size_bytes) for url in urls if url in metadata_by_url)
        periods = [archive_url_period(url) for url in urls]
        archive_plans.append({
            "symbol": symbol,
            "listed_files": int(listed_count),
            "listed_bytes": int(listed_bytes),
            "filtered_files": int(filtered_count),
            "filtered_bytes": int(filtered_bytes),
            "selected_files": int(len(urls)),
            "selected_bytes": int(selected_bytes),
            "size_metadata_available": bool(metadata_by_url),
            "first_period": next((period for period in periods if period), ""),
            "last_period": next((period for period in reversed(periods) if period), ""),
            "first_url": urls[0] if urls else "",
            "last_url": urls[-1] if urls else "",
        })
        if max_files_int == 0 and filtered_count > 0:
            continue
        if not urls:
            window_suffix = "_in_period_window" if start_period or end_period else ""
            errors.append({"symbol": symbol, "error": f"no_{cadence}_archive_files{window_suffix}"})
            continue
        selected_archive_urls[symbol] = list(urls)
    shortfall = requested_top_symbols > 0 and len(symbols) < requested_top_symbols
    planned_files = sum(int(item["selected_files"]) for item in archive_plans)
    planned_bytes = sum(int(item["selected_bytes"]) for item in archive_plans)
    max_planned_gb = max(0.0, float(getattr(args, "max_planned_gb", 50.0) or 0.0))
    max_planned_bytes = int(max_planned_gb * 1_000_000_000)
    if (
        not plan_only
        and max_planned_bytes > 0
        and planned_bytes > max_planned_bytes
    ):
        errors.append({
            "symbol": "*",
            "error": f"planned_bytes_exceeds_max:{planned_bytes}/{max_planned_bytes}",
        })
    if not plan_only and not errors and not shortfall:
        for symbol, urls in selected_archive_urls.items():
            all_results.extend(
                ingest_archive_urls(
                    db_path=Path(getattr(args, "db", "data/market_data.sqlite")),
                    symbol=symbol,
                    interval=interval,
                    urls=urls,
                    market_type=market_type,
                    data_type=data_type,
                    timeout=max(1, int(getattr(args, "timeout", 120) or 120)),
                    force=bool(getattr(args, "force", False)),
                    verify_checksum=not bool(getattr(args, "no_verify_checksum", False)),
                    require_checksum=bool(getattr(args, "require_checksum", False)),
                    store_raw_agg_trades=not bool(getattr(args, "aggregate_only", False)),
                )
            )
    payload = {
        "status": (
            "ok"
            if not errors and not shortfall and all(item.status in {"complete", "skipped"} for item in all_results)
            else "warn"
        ),
        "symbol": symbols[0] if len(symbols) == 1 else "",
        "symbols": symbols,
        "symbol_count": len(symbols),
        "requested_top_symbols": int(requested_top_symbols),
        "interval": interval,
        "data_type": data_type,
        "market_type": market_type,
        "cadence": cadence,
        "plan_only": bool(plan_only),
        "start_period": start_period or "",
        "end_period": end_period or "",
        "max_planned_gb": float(max_planned_gb),
        "max_planned_bytes": int(max_planned_bytes),
        "files": len(all_results),
        "planned_files": int(planned_files),
        "planned_bytes": int(planned_bytes),
        "rows_read": sum(item.rows_read for item in all_results),
        "rows_inserted": sum(item.rows_inserted for item in all_results),
        "bytes_downloaded": sum(item.bytes_downloaded for item in all_results),
        "archive_plans": archive_plans,
        "history_rejections": history_rejections,
        "errors": errors,
        "results": [item.asdict() for item in all_results],
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "archive-sync: "
            f"status={payload['status']} symbols={payload['symbol_count']} interval={interval} data_type={data_type} market={market_type} "
            f"planned_files={payload['planned_files']} planned_bytes={payload['planned_bytes']} files={payload['files']} rows_read={payload['rows_read']} "
            f"rows_inserted={payload['rows_inserted']} bytes={payload['bytes_downloaded']}"
        )
        if plan_only:
            print("archive-sync plan-only: no files downloaded or ingested")
        if shortfall:
            print(
                f"warning: archive-sync selected {len(symbols)}/{requested_top_symbols} symbols after history-depth gates",
                file=sys.stderr,
            )
        for error in [*history_rejections, *errors]:
            print(f"warning: {error['symbol']} {error['error']}", file=sys.stderr)
        for item in all_results:
            if item.status == "error":
                print(f"warning: {item.url} {item.error}", file=sys.stderr)
    return 0 if payload["status"] == "ok" else 2


def command_microstructure_capture(args: argparse.Namespace) -> int:
    raw_symbols = str(getattr(args, "symbols", "") or "")
    symbols = tuple(item.strip().upper() for item in raw_symbols.split(",") if item.strip())
    if not symbols:
        symbols = tuple(load_runtime().symbols)
    convert = bool(getattr(args, "convert", True))
    json_mode = bool(getattr(args, "json", False))
    if convert and importlib.util.find_spec("hftbacktest") is None:
        print(
            "microstructure-capture requires the microstructure extra: "
            "python -m pip install -e .[microstructure]",
            file=sys.stderr,
        )
        return 2
    try:
        def progress(completed: float, total: float) -> None:
            if not json_mode:
                print(
                    "microstructure-capture stream: "
                    f"{completed:.0f}/{total:.0f}s",
                    flush=True,
                )

        result = capture_binance_futures_microstructure(
            symbols,
            duration_seconds=float(getattr(args, "seconds", 60.0)),
            output_root=str(getattr(args, "output_root", "data/microstructure")),
            timeout_seconds=float(getattr(args, "timeout", 10.0)),
            convert=convert,
            progress=progress,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"microstructure-capture failed: {exc}", file=sys.stderr)
        return 2
    catalog_changes = 0
    try:
        with MarketDataStore(str(getattr(args, "db", "data/market_data.sqlite"))) as store:
            catalog_changes = store.record_microstructure_capture(result.asdict())
    except (OSError, ValueError) as exc:
        print(f"microstructure-capture catalog registration failed: {exc}", file=sys.stderr)
        return 2
    payload = result.asdict()
    payload["catalog_changes"] = int(catalog_changes)
    payload["catalog_db"] = str(getattr(args, "db", "data/market_data.sqlite"))
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "microstructure-capture: "
            f"status={result.status} capture_id={result.capture_id} "
            f"symbols={len(result.evidence)} catalog_changes={catalog_changes}"
        )
        for item in result.evidence:
            print(
                f"  {item.symbol}: raw_messages={item.raw_messages} depth_messages={item.depth_messages} "
                f"trades={item.trade_messages} bbo={item.book_ticker_messages} "
                f"gaps={item.sequence_gap_count} invalid={item.invalid_event_count} "
                f"normalized_rows={item.normalized_rows} replay={item.replay_smoke_passed}"
            )
            if item.error:
                print(f"    error={item.error}", file=sys.stderr)
        if result.errors:
            for error in result.errors:
                print(f"warning: {error}", file=sys.stderr)
        print(f"manifest={result.manifest_path}")
    return 0 if result.status == "pass" else 2


def _parse_cli_utc_date(value: object, label: str):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD") from exc


def command_tick_archive_sync(args: argparse.Namespace) -> int:
    import duckdb
    import requests

    from .microstructure_warehouse import (
        MicrostructureWarehouse,
        SUPPORTED_TICK_ARCHIVES,
        TickArchiveIngestResult,
    )

    recoverable_errors = (OSError, RuntimeError, ValueError, requests.RequestException, duckdb.Error)

    raw_symbols = str(getattr(args, "symbols", "") or "")
    symbols = tuple(item.strip().upper() for item in raw_symbols.split(",") if item.strip())
    if not symbols:
        symbols = tuple(load_runtime().symbols)
    data_types = tuple(
        item.strip() for item in str(getattr(args, "data_types", "") or "").split(",") if item.strip()
    )
    unsupported = sorted(set(data_types) - set(SUPPORTED_TICK_ARCHIVES))
    if not data_types or unsupported:
        detail = ",".join(unsupported) if unsupported else "missing"
        print(f"tick-archive-sync unsupported data types: {detail}", file=sys.stderr)
        return 2
    full_history = bool(getattr(args, "full_history", False))
    available_only = bool(getattr(args, "available_only", False)) or full_history
    plan_only = bool(getattr(args, "plan_only", False))
    start_value = getattr(args, "start_date", None)
    end_value = getattr(args, "end_date", None)
    if full_history and (start_value or end_value):
        print(
            "tick-archive-sync: --full-history cannot be combined with "
            "--start-date or --end-date",
            file=sys.stderr,
        )
        return 2
    if not full_history and (not start_value or not end_value):
        print(
            "tick-archive-sync: provide both --start-date and --end-date or "
            "use --full-history",
            file=sys.stderr,
        )
        return 2
    start = None
    end = None
    requested_periods: set[str] | None = None
    if not full_history:
        try:
            start = _parse_cli_utc_date(start_value, "start-date")
            end = _parse_cli_utc_date(end_value, "end-date")
        except ValueError as exc:
            print(f"tick-archive-sync: {exc}", file=sys.stderr)
            return 2
        if start > end:
            print("tick-archive-sync: start-date must not be after end-date", file=sys.stderr)
            return 2
        requested_periods = {
            (start + timedelta(days=offset)).isoformat()
            for offset in range((end - start).days + 1)
        }
    plan: list[tuple[str, str, object]] = []
    missing: list[dict[str, str]] = []
    coverage: list[dict[str, object]] = []
    try:
        for symbol in symbols:
            for data_type in data_types:
                items = list_archive_items(
                    symbol=symbol,
                    interval="tick",
                    market_type="futures",
                    cadence="daily",
                    data_type=data_type,
                    timeout=max(1, min(60, int(float(getattr(args, "timeout", 240.0))))),
                )
                incomplete_object_metadata = [
                    item.period
                    for item in items
                    if (
                        int(getattr(item, "size_bytes", 0) or 0) <= 0
                        or not str(getattr(item, "last_modified", "") or "")
                        or not str(getattr(item, "etag", "") or "")
                        or int(getattr(item, "checksum_size_bytes", 0) or 0) <= 0
                        or not str(
                            getattr(item, "checksum_last_modified", "") or ""
                        )
                        or not str(getattr(item, "checksum_etag", "") or "")
                    )
                ]
                if incomplete_object_metadata:
                    raise ValueError(
                        f"{symbol} {data_type} official listing lacks ZIP/CHECKSUM "
                        f"metadata for {incomplete_object_metadata[0]}"
                    )
                by_period = {item.period: item for item in items}
                official_calendar_gaps: list[str] = []
                if by_period:
                    calendar_cursor = datetime.strptime(
                        min(by_period), "%Y-%m-%d"
                    ).date()
                    calendar_end = datetime.strptime(
                        max(by_period), "%Y-%m-%d"
                    ).date()
                    while calendar_cursor <= calendar_end:
                        calendar_period = calendar_cursor.isoformat()
                        if calendar_period not in by_period:
                            official_calendar_gaps.append(calendar_period)
                        calendar_cursor += timedelta(days=1)
                selected_periods = (
                    sorted(by_period)
                    if requested_periods is None
                    else sorted(requested_periods)
                )
                selected_items = []
                for period in selected_periods:
                    item = by_period.get(period)
                    if item is None:
                        missing.append(
                            {
                                "symbol": symbol,
                                "data_type": data_type,
                                "period": period,
                            }
                        )
                    else:
                        selected_items.append(item)
                        plan.append((symbol, data_type, item))
                coverage.append(
                    {
                        "symbol": symbol,
                        "data_type": data_type,
                        "available_files": len(items),
                        "available_first_period": min(by_period) if by_period else None,
                        "available_last_period": max(by_period) if by_period else None,
                        "official_calendar_gap_count": len(official_calendar_gaps),
                        "official_calendar_gaps": official_calendar_gaps,
                        "selected_files": len(selected_items),
                        "selected_first_period": (
                            selected_items[0].period if selected_items else None
                        ),
                        "selected_last_period": (
                            selected_items[-1].period if selected_items else None
                        ),
                        "selected_bytes": sum(
                            int(item.size_bytes) for item in selected_items
                        ),
                    }
                )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"tick-archive-sync planning failed: {exc}", file=sys.stderr)
        return 2
    plan.sort(key=lambda value: (value[0], value[1], value[2].period))
    json_mode = bool(getattr(args, "json", False))
    planned_bytes = sum(int(item.size_bytes) for _, _, item in plan)
    max_planned_gb = float(getattr(args, "max_planned_gb", 500.0))
    if not math.isfinite(max_planned_gb) or max_planned_gb < 0.0:
        print("tick-archive-sync: --max-planned-gb must be finite and non-negative", file=sys.stderr)
        return 2
    plan_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "truth_basis": "official_binance_data_vision_s3_listing",
        "status": (
            "ok"
            if plan and (available_only or not missing)
            else "incomplete"
        ),
        "plan_only": plan_only,
        "full_history": full_history,
        "available_only": available_only,
        "warehouse": str(getattr(args, "warehouse", "data/microstructure.duckdb")),
        "symbols": list(symbols),
        "data_types": list(data_types),
        "start_date": start.isoformat() if start is not None else None,
        "end_date": end.isoformat() if end is not None else None,
        "planned_files": len(plan),
        "planned_bytes": planned_bytes,
        "planned_gb": planned_bytes / 1024**3,
        "max_planned_gb": max_planned_gb,
        "coverage": coverage,
        "missing": missing,
    }
    plan_output = getattr(args, "plan_output", None)
    if plan_only:
        plan_inventory_snapshots: list[dict[str, object]] = []
        inventory_errors: list[dict[str, str]] = []
        if full_history:
            try:
                with MicrostructureWarehouse(
                    str(getattr(args, "warehouse", "data/microstructure.duckdb")),
                    cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
                    memory_limit=str(getattr(args, "memory_limit", "8GB")),
                    threads=int(getattr(args, "threads", 8)),
                ) as warehouse:
                    inventory_groups: dict[tuple[str, str], list[object]] = {}
                    for symbol, data_type, item in plan:
                        inventory_groups.setdefault((symbol, data_type), []).append(item)
                    initial_ids: dict[tuple[str, str], str] = {}
                    for (symbol, data_type), items in sorted(inventory_groups.items()):
                        snapshot = warehouse.record_official_archive_inventory(
                            symbol=symbol,
                            data_type=data_type,
                            items=items,
                            full_history=True,
                        )
                        plan_inventory_snapshots.append(
                            {**snapshot, "verification_phase": "initial"}
                        )
                        initial_ids[(symbol, data_type)] = str(snapshot["snapshot_id"])
                    for symbol, data_type in sorted(inventory_groups):
                        refreshed_items = list_archive_items(
                            symbol=symbol,
                            interval="tick",
                            market_type="futures",
                            cadence="daily",
                            data_type=data_type,
                            timeout=max(
                                1,
                                min(
                                    60,
                                    int(float(getattr(args, "timeout", 240.0))),
                                ),
                            ),
                        )
                        refreshed = warehouse.record_official_archive_inventory(
                            symbol=symbol,
                            data_type=data_type,
                            items=refreshed_items,
                            full_history=True,
                        )
                        plan_inventory_snapshots.append(
                            {
                                **refreshed,
                                "verification_phase": "verification_refresh",
                            }
                        )
                        if str(refreshed["snapshot_id"]) != initial_ids.get(
                            (symbol, data_type)
                        ):
                            inventory_errors.append(
                                {
                                    "symbol": symbol,
                                    "data_type": data_type,
                                    "error": "official_inventory_changed_during_plan",
                                }
                            )
            except recoverable_errors as exc:
                print(
                    f"tick-archive-sync inventory plan failed: {exc}",
                    file=sys.stderr,
                )
                return 2
        plan_payload["inventory_snapshots"] = plan_inventory_snapshots
        plan_payload["inventory_identity_verified"] = bool(
            full_history and plan_inventory_snapshots and not inventory_errors
        )
        plan_payload["inventory_errors"] = inventory_errors
        if inventory_errors:
            plan_payload["status"] = "incomplete"
        if plan_output:
            write_json_atomic(
                Path(str(plan_output)),
                plan_payload,
                indent=2,
                sort_keys=True,
            )
        if json_mode:
            print(json.dumps(plan_payload, indent=2, sort_keys=True))
        else:
            print(
                "tick-archive-sync plan: "
                f"status={plan_payload['status']} files={len(plan)} "
                f"compressed_gb={float(plan_payload['planned_gb']):.3f} "
                f"missing={len(missing)}"
            )
            for item in coverage:
                print(
                    f"  {item['symbol']} {item['data_type']}: "
                    f"{item['selected_first_period']}..{item['selected_last_period']} "
                    f"files={item['selected_files']} "
                    f"provider_gaps={item['official_calendar_gap_count']} "
                    f"compressed_gb={int(item['selected_bytes']) / 1024**3:.3f}"
                )
        return 0 if plan_payload["status"] == "ok" else 2
    if plan_output:
        write_json_atomic(
            Path(str(plan_output)),
            plan_payload,
            indent=2,
            sort_keys=True,
        )
    if max_planned_gb > 0.0 and planned_bytes > max_planned_gb * 1024**3:
        print(
            "tick-archive-sync: official compressed-byte plan exceeds "
            f"--max-planned-gb ({planned_bytes / 1024**3:.3f} > {max_planned_gb:.3f}); "
            "run --plan-only, raise the limit, or use 0 to disable",
            file=sys.stderr,
        )
        return 2
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    inventory_snapshots: list[dict[str, object]] = []
    initial_inventory_ids: dict[tuple[str, str], str] = {}
    corpus_certificates: list[dict[str, object]] = []
    try:
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            inventory_groups: dict[tuple[str, str], list[object]] = {}
            reusable_archives: dict[
                tuple[str, str, str], TickArchiveIngestResult
            ] = {}
            for symbol, data_type, item in plan:
                inventory_groups.setdefault((symbol, data_type), []).append(item)
            for (symbol, data_type), items in sorted(inventory_groups.items()):
                snapshot = warehouse.record_official_archive_inventory(
                    symbol=symbol,
                    data_type=data_type,
                    items=items,
                    full_history=full_history,
                    scope_start_period=(start.isoformat() if start is not None else None),
                    scope_end_period=(end.isoformat() if end is not None else None),
                )
                inventory_snapshots.append(
                    {**snapshot, "verification_phase": "initial"}
                )
                initial_inventory_ids[(symbol, data_type)] = str(snapshot["snapshot_id"])
                reusable_archives.update(
                    {
                        (symbol, data_type, period): result
                        for period, result in warehouse.reusable_official_archives(
                            symbol=symbol,
                            data_type=data_type,
                            items=items,
                        ).items()
                    }
                )
            for symbol, data_type, item in plan:
                if not json_mode:
                    print(f"tick-archive-sync start {symbol} {data_type} {item.period}", flush=True)

                reusable = reusable_archives.get((symbol, data_type, item.period))
                if reusable is not None:
                    results.append(reusable.asdict())
                    if not json_mode:
                        print(
                            f"tick-archive-sync complete {symbol} {data_type} {item.period}: "
                            f"status={reusable.status} rows={reusable.rows_read}",
                            flush=True,
                        )
                    continue

                def progress(phase: str, completed: int, total: int | None) -> None:
                    if not json_mode:
                        print(
                            f"  {phase}: {completed}/{total if total is not None else '?'}",
                            flush=True,
                        )

                try:
                    result = warehouse.ingest_public_archive(
                        symbol=symbol,
                        data_type=data_type,
                        period=item.period,
                        url=item.url,
                        expected_bytes=item.size_bytes,
                        official_last_modified=item.last_modified,
                        official_etag=item.etag,
                        checksum_object_size_bytes=item.checksum_size_bytes,
                        checksum_last_modified=item.checksum_last_modified,
                        checksum_etag=item.checksum_etag,
                        timeout_seconds=float(getattr(args, "timeout", 240.0)),
                        retain_archive=not bool(getattr(args, "no_retain_archive", False)),
                        progress=progress,
                    )
                    results.append(result.asdict())
                except recoverable_errors as exc:
                    errors.append(
                        {
                            "symbol": symbol,
                            "data_type": data_type,
                            "period": item.period,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if not json_mode:
                        print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                if not json_mode:
                    print(
                        f"tick-archive-sync complete {symbol} {data_type} {item.period}: "
                        f"status={result.status} rows={result.rows_read}",
                        flush=True,
                    )
            if full_history:
                for symbol in symbols:
                    for data_type in data_types:
                        refreshed_items = list_archive_items(
                            symbol=symbol,
                            interval="tick",
                            market_type="futures",
                            cadence="daily",
                            data_type=data_type,
                            timeout=max(
                                1,
                                min(
                                    60,
                                    int(float(getattr(args, "timeout", 240.0))),
                                ),
                            ),
                        )
                        refreshed = warehouse.record_official_archive_inventory(
                            symbol=symbol,
                            data_type=data_type,
                            items=refreshed_items,
                            full_history=True,
                        )
                        inventory_snapshots.append(
                            {
                                **refreshed,
                                "verification_phase": "verification_refresh",
                            }
                        )
                        if str(refreshed["snapshot_id"]) != initial_inventory_ids.get(
                            (symbol, data_type)
                        ):
                            errors.append(
                                {
                                    "symbol": symbol,
                                    "data_type": data_type,
                                    "period": "full_history",
                                    "error": "official_inventory_changed_during_sync",
                                }
                            )
                for symbol in symbols:
                    certificate = warehouse.corpus_certificate(
                        symbol,
                        required_data_types=data_types,
                        allow_official_gap_data_types=(
                            ("bookDepth",) if "bookDepth" in data_types else ()
                        ),
                    )
                    corpus_certificates.append(certificate)
                    if certificate["status"] != "pass":
                        errors.append(
                            {
                                "symbol": symbol,
                                "data_type": "corpus_certificate",
                                "period": "full_history",
                                "error": "; ".join(
                                    str(value)
                                    for value in certificate.get("reasons", [])[:8]
                                ),
                            }
                        )
    except recoverable_errors as exc:
        print(f"tick-archive-sync warehouse failed: {exc}", file=sys.stderr)
        return 2
    payload = {
        **plan_payload,
        "status": (
            "ok"
            if plan and not errors and (available_only or not missing)
            else "incomplete"
        ),
        "completed_files": len(results),
        "reused_files": sum(
            item.get("status") == "skipped_verified_unchanged" for item in results
        ),
        "ingested_files": sum(item.get("status") == "complete" for item in results),
        "inventory_snapshots": inventory_snapshots,
        "corpus_certificates": corpus_certificates,
        "errors": errors,
        "results": results,
    }
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "tick-archive-sync: "
            f"status={payload['status']} planned={len(plan)} completed={len(results)} "
            f"reused={payload['reused_files']} ingested={payload['ingested_files']} "
            f"missing={len(missing)} errors={len(errors)}"
        )
    return 0 if payload["status"] == "ok" else 2


def command_tick_corpus_audit(args: argparse.Namespace) -> int:
    """Emit a fail-closed certificate for the reusable full-history corpus."""

    import duckdb

    from .assets import normalize_symbols
    from .microstructure_warehouse import MicrostructureWarehouse, SUPPORTED_TICK_ARCHIVES

    symbols = normalize_symbols(
        str(getattr(args, "symbols", "BTCUSDT,ETHUSDT,SOLUSDT")),
        default=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    )
    data_types = tuple(
        dict.fromkeys(
            item.strip()
            for item in str(
                getattr(args, "data_types", "bookTicker,trades,bookDepth")
            ).split(",")
            if item.strip()
        )
    )
    if not data_types or any(item not in SUPPORTED_TICK_ARCHIVES for item in data_types):
        print("tick-corpus-audit: unsupported or missing data type", file=sys.stderr)
        return 2
    start_value = getattr(args, "start_date", None)
    end_value = getattr(args, "end_date", None)
    if bool(start_value) != bool(end_value):
        print(
            "tick-corpus-audit: provide both --start-date and --end-date",
            file=sys.stderr,
        )
        return 2
    start_ms: int | None = None
    end_ms: int | None = None
    if start_value and end_value:
        try:
            start = _parse_cli_utc_date(start_value, "start-date")
            end = _parse_cli_utc_date(end_value, "end-date")
        except ValueError as exc:
            print(f"tick-corpus-audit: {exc}", file=sys.stderr)
            return 2
        if start > end:
            print(
                "tick-corpus-audit: start-date must not be after end-date",
                file=sys.stderr,
            )
            return 2
        start_ms = int(
            datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp()
            * 1_000
        )
        end_ms = int(
            datetime.combine(
                end + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).timestamp()
            * 1_000
        ) - 1
    try:
        allowed_official_gap_types = (
            ("bookDepth",)
            if "bookDepth" in data_types
            and bool(getattr(args, "allow_provider_book_depth_gaps", True))
            else ()
        )
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            certificates = [
                warehouse.corpus_certificate(
                    symbol,
                    required_data_types=data_types,
                    required_start_ms=start_ms,
                    required_end_ms=end_ms,
                    allow_official_gap_data_types=allowed_official_gap_types,
                )
                for symbol in symbols
            ]
    except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
        print(f"tick-corpus-audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    payload = {
        "contract": "official-binance-multi-symbol-corpus-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "pass"
            if certificates and all(item["status"] == "pass" for item in certificates)
            else "fail"
        ),
        "warehouse": str(getattr(args, "warehouse", "data/microstructure.duckdb")),
        "symbols": list(symbols),
        "required_data_types": list(data_types),
        "allowed_official_gap_data_types": list(allowed_official_gap_types),
        "required_start_ms": start_ms,
        "required_end_ms": end_ms,
        "certificates": certificates,
    }
    output = getattr(args, "output", None)
    if output:
        write_json_atomic(Path(str(output)), payload, indent=2, sort_keys=True)
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"tick-corpus-audit: status={payload['status']} symbols={len(certificates)}"
        )
        for item in certificates:
            print(
                f"  {item['symbol']}: {item['status']} "
                f"common={item['common_first_period']}..{item['common_last_period']} "
                f"days={item['common_period_count']}"
            )
            if item["reasons"]:
                print("    " + "; ".join(str(reason) for reason in item["reasons"][:8]))
    return 0 if payload["status"] == "pass" else 2


def command_microstructure_train(args: argparse.Namespace) -> int:
    from .microstructure_features import (
        apply_path_aware_lifecycle_targets,
        build_executable_microstructure_dataset,
    )
    from .microstructure_model import (
        save_microstructure_model_artifact,
        train_microstructure_action_value_model,
    )
    from .microstructure_warehouse import MicrostructureWarehouse

    json_mode = bool(getattr(args, "json", False))
    stop = getattr(args, "stop_loss_bps", None)
    take = getattr(args, "take_profit_bps", None)
    if (stop is None) != (take is None):
        print("microstructure-train requires both --stop-loss-bps and --take-profit-bps", file=sys.stderr)
        return 2
    evaluate_terminal = bool(getattr(args, "evaluate_terminal", False))
    risk_level = str(getattr(args, "risk_level", "conservative"))
    configured_participation = getattr(args, "max_l1_participation", None)
    max_l1_participation = (
        {"conservative": 0.05, "regular": 0.10, "aggressive": 0.15}[risk_level]
        if configured_participation is None
        else float(configured_participation)
    )
    output = str(getattr(args, "output", "data/microstructure-model.json"))
    if evaluate_terminal and (stop is None or take is None):
        print(
            "microstructure-train --evaluate-terminal requires explicit path-aware "
            "--stop-loss-bps and --take-profit-bps",
            file=sys.stderr,
        )
        return 2
    if evaluate_terminal:
        print(
            "microstructure-train --evaluate-terminal is disabled: train the exact "
            "candidate, run microstructure-prequential, then use "
            "microstructure-promote with the hash-bound evidence",
            file=sys.stderr,
        )
        return 2

    def model_progress(phase: str, completed: int, total: int) -> None:
        if not json_mode:
            print(f"microstructure-train {phase}: {completed}/{total}", flush=True)

    def warehouse_progress(phase: str, completed: int, total: int | None) -> None:
        if not json_mode:
            print(f"microstructure-train {phase}: {completed}/{total}", flush=True)

    try:
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            backfilled = warehouse.backfill_book_ticker_paths(progress=warehouse_progress)
            if not json_mode and backfilled:
                print(f"microstructure-train backfilled_archives={len(backfilled)}", flush=True)
            causal_evidence = warehouse.rebuild_causal_feature_bars(
                str(getattr(args, "symbol", "BTCUSDT")),
                progress=warehouse_progress,
            )
            if not json_mode:
                print(
                    "microstructure-train causal_feature_bars="
                    f"{causal_evidence['feature_rows']} "
                    f"source_rows={causal_evidence['source_raw_rows']} "
                    f"build={str(causal_evidence['build_id'])[:12]}",
                    flush=True,
                )
            dataset = build_executable_microstructure_dataset(
                warehouse,
                symbol=str(getattr(args, "symbol", "BTCUSDT")),
                horizon_seconds=int(getattr(args, "horizon_seconds", 900)),
                total_latency_ms=int(getattr(args, "total_latency_ms", 750)),
                taker_fee_bps=float(getattr(args, "taker_fee_bps", 5.0)),
                additional_slippage_bps_per_side=float(
                    getattr(args, "additional_slippage_bps_per_side", 1.0)
                ),
                max_quote_age_ms=int(getattr(args, "max_quote_age_ms", 1000)),
                reference_order_notional_quote=float(
                    getattr(args, "reference_order_notional_quote", 1_000.0)
                ),
                max_l1_participation=max_l1_participation,
                decision_cadence_seconds=int(
                    getattr(args, "decision_cadence_seconds", 5)
                ),
            )
            path_evidence = None
            if stop is not None and take is not None:
                dataset, path_evidence = apply_path_aware_lifecycle_targets(
                    warehouse,
                    dataset,
                    stop_loss_bps=float(stop),
                    take_profit_bps=float(take),
                    trigger_execution_slippage_bps=float(
                        getattr(args, "trigger_slippage_bps", 1.0)
                    ),
                )
            artifact = train_microstructure_action_value_model(
                dataset,
                risk_level=risk_level,
                compute_backend=str(getattr(args, "compute_backend", "auto")),
                minimum_promotion_days=int(getattr(args, "minimum_promotion_days", 240)),
                deployment_calibration_days=int(
                    getattr(args, "deployment_calibration_days", 14)
                ),
                maximum_model_age_seconds=int(
                    getattr(args, "maximum_model_age_seconds", 86_400)
                ),
                evaluate_terminal=False,
                progress=model_progress,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"microstructure-train failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    digest = save_microstructure_model_artifact(artifact, output)
    payload = artifact.asdict()
    payload.pop("model_strings", None)
    payload.pop("deployment_model_strings", None)
    payload["artifact_path"] = output
    payload["artifact_sha256"] = digest
    payload["path_target_evidence"] = asdict(path_evidence) if path_evidence is not None else None
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "microstructure-train: "
            f"status={artifact.status} rows={dataset.rows} days={artifact.unique_utc_days} "
            f"policy_trades={artifact.policy_metrics.trades} "
            f"selection_trades={artifact.selection_metrics.trades} "
            f"selection_net_bps={artifact.selection_metrics.total_net_bps:+.4f} "
            f"selection_daily_ci95=["
            f"{artifact.selection_confidence.mean_daily_net_bps_ci_lower:+.4f},"
            f"{artifact.selection_confidence.mean_daily_net_bps_ci_upper:+.4f}] "
            f"l1_eligible_long={float(artifact.dataset_summary['long_liquidity_eligible_ratio'] or 0.0):.3f} "
            f"l1_eligible_short={float(artifact.dataset_summary['short_liquidity_eligible_ratio'] or 0.0):.3f}"
        )
        if artifact.terminal_metrics is not None:
            terminal_confidence = artifact.terminal_confidence
            terminal_ci = (
                "unavailable"
                if terminal_confidence is None
                else "["
                f"{terminal_confidence.mean_daily_net_bps_ci_lower:+.4f},"
                f"{terminal_confidence.mean_daily_net_bps_ci_upper:+.4f}]"
            )
            print(
                "terminal: "
                f"trades={artifact.terminal_metrics.trades} "
                f"net_bps={artifact.terminal_metrics.total_net_bps:+.4f} "
                f"daily_ci95={terminal_ci}"
            )
        if artifact.rejection_reasons:
            print("rejected: " + "; ".join(artifact.rejection_reasons), file=sys.stderr)
        print(f"artifact={output} sha256={digest}")
    return 0 if artifact.status == "candidate" else 2


def command_microstructure_refit(args: argparse.Namespace) -> int:
    from .microstructure_features import (
        apply_path_aware_lifecycle_targets,
        build_executable_microstructure_dataset,
    )
    from .microstructure_model import (
        load_microstructure_model_artifact,
        refit_validated_microstructure_model,
        save_microstructure_model_artifact,
    )
    from .microstructure_warehouse import MicrostructureWarehouse

    input_path = Path(str(getattr(args, "input", "data/microstructure-model.json")))
    output_value = getattr(args, "output", None)
    output_path = Path(str(output_value)) if output_value else input_path
    json_mode = bool(getattr(args, "json", False))

    def progress(phase: str, completed: int, total: int) -> None:
        if not json_mode:
            print(f"microstructure-refit {phase}: {completed}/{total}", flush=True)

    try:
        artifact = load_microstructure_model_artifact(input_path)
        if artifact.status != "validated":
            raise ValueError("microstructure-refit requires a terminal-validated artifact")
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            warehouse.require_causal_feature_bars(artifact.symbol)
            dataset = build_executable_microstructure_dataset(
                warehouse,
                symbol=artifact.symbol,
                horizon_seconds=artifact.horizon_seconds,
                total_latency_ms=artifact.total_latency_ms,
                taker_fee_bps=artifact.taker_fee_bps,
                additional_slippage_bps_per_side=(
                    artifact.additional_slippage_bps_per_side
                ),
                max_quote_age_ms=artifact.max_quote_age_ms,
                reference_order_notional_quote=artifact.reference_order_notional_quote,
                max_l1_participation=artifact.max_l1_participation,
                decision_cadence_seconds=artifact.decision_cadence_seconds,
            )
            dataset, _path_evidence = apply_path_aware_lifecycle_targets(
                warehouse,
                dataset,
                stop_loss_bps=float(artifact.stop_loss_bps or 0.0),
                take_profit_bps=float(artifact.take_profit_bps or 0.0),
                trigger_execution_slippage_bps=float(
                    artifact.trigger_execution_slippage_bps or 0.0
                ),
            )
            shadow_candidate = refit_validated_microstructure_model(
                artifact,
                dataset,
                compute_backend=str(getattr(args, "compute_backend", "auto")),
                progress=progress,
            )
        digest = save_microstructure_model_artifact(shadow_candidate, output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"microstructure-refit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    payload = {
        "status": shadow_candidate.status,
        "trading_authority": False,
        "shadow_required": True,
        "artifact_path": str(output_path),
        "artifact_sha256": digest,
        "symbol": shadow_candidate.symbol,
        "deployment_refit": (
            asdict(shadow_candidate.deployment_refit)
            if shadow_candidate.deployment_refit is not None
            else None
        ),
    }
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        refit = shadow_candidate.deployment_refit
        print(
            "microstructure-refit: "
            f"status={shadow_candidate.status} symbol={shadow_candidate.symbol} "
            f"cutoff_ms={refit.training_cutoff_ms if refit else 'n/a'} "
            f"expires_at_ms={refit.expires_at_ms if refit else 'n/a'} "
            "trading_authority=false shadow_required=true"
        )
        print(f"artifact={output_path} sha256={digest}")
    return 0 if shadow_candidate.status == "shadow_candidate" else 2


def command_microstructure_shadow(args: argparse.Namespace) -> int:
    """Capture public market data and run the locked no-order promotion gate."""

    from .microstructure_model import (
        load_microstructure_model_artifact,
        save_microstructure_model_artifact,
    )
    from .microstructure_shadow import (
        PROMOTION_SHADOW_CONFIG,
        evaluate_shadow_capture,
    )

    input_path = Path(str(getattr(args, "input", "data/microstructure-model.json")))
    output_value = getattr(args, "output", None)
    output_path = Path(str(output_value)) if output_value else input_path
    report_path = Path(
        str(getattr(args, "report", "data/microstructure-shadow/report.json"))
    )
    trades_path = Path(
        str(getattr(args, "trades", "data/microstructure-shadow/trades.csv"))
    )
    duration_seconds = float(
        getattr(
            args,
            "seconds",
            21_600.0 + MICROSTRUCTURE_STREAM_WARMUP_SECONDS + 60.0,
        )
    )
    json_mode = bool(getattr(args, "json", False))
    catalog_changes = 0

    def progress(completed: float, total: float) -> None:
        if not json_mode:
            print(
                f"microstructure-shadow capture: {completed:.0f}/{total:.0f}s",
                flush=True,
            )

    try:
        artifact = load_microstructure_model_artifact(input_path)
        if artifact.status != "shadow_candidate" or artifact.rejection_reasons:
            raise ValueError(
                "microstructure-shadow requires an unrejected shadow_candidate"
            )
        if artifact.deployment_refit is None:
            raise ValueError("microstructure-shadow candidate has no deployment refit")
        minimum_capture = (
            PROMOTION_SHADOW_CONFIG.minimum_duration_seconds
            + MICROSTRUCTURE_STREAM_WARMUP_SECONDS
            + 60.0
        )
        if not math.isfinite(duration_seconds) or duration_seconds < minimum_capture:
            raise ValueError(
                f"microstructure-shadow requires at least {minimum_capture:.0f} "
                "seconds so feature warmup is followed by six complete evaluated hours"
            )
        now_ms = int(time.time() * 1_000)
        required_lifetime_ms = int(
            (
                duration_seconds
                + PROMOTION_SHADOW_CONFIG.maximum_capture_age_seconds
                + 120.0
            )
            * 1_000
        )
        if artifact.deployment_refit.expires_at_ms - now_ms < required_lifetime_ms:
            raise ValueError(
                "deployment refit will expire before the shadow capture and evaluation "
                "can complete; rebuild current features and refit first"
            )
        capture = capture_binance_futures_microstructure(
            (artifact.symbol,),
            duration_seconds=duration_seconds,
            output_root=str(
                getattr(args, "output_root", "data/microstructure-shadow/captures")
            ),
            timeout_seconds=float(getattr(args, "timeout", 10.0)),
            convert=False,
            progress=progress,
        )
        with MarketDataStore(
            str(getattr(args, "db", "data/market_data.sqlite"))
        ) as store:
            catalog_changes = store.record_microstructure_capture(capture.asdict())
        report, accepted = evaluate_shadow_capture(
            artifact,
            input_path,
            capture,
            report_path=report_path,
            trades_path=trades_path,
            config=PROMOTION_SHADOW_CONFIG,
        )
        digest = (
            save_microstructure_model_artifact(accepted, output_path)
            if accepted is not None
            else None
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"microstructure-shadow failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    payload = report.asdict()
    payload.update(
        {
            "artifact_path": str(output_path),
            "artifact_sha256": digest,
            "catalog_changes": int(catalog_changes),
            "catalog_db": str(getattr(args, "db", "data/market_data.sqlite")),
            "report_path": str(report_path),
        }
    )
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "microstructure-shadow: "
            f"status={report.status} symbol={report.symbol} "
            f"decisions={report.replay['decisions']} "
            f"trades={report.metrics['trades']} "
            f"net_bps={float(report.metrics['total_net_bps']):+.4f} "
            f"trading_authority={str(report.trading_authority).lower()}"
        )
        if report.reasons:
            print("rejected: " + "; ".join(report.reasons), file=sys.stderr)
        print(f"report={report_path} trades={trades_path}")
        if digest is not None:
            print(f"artifact={output_path} sha256={digest}")
    return 0 if accepted is not None and report.passed else 2


def command_tape_depth_train(args: argparse.Namespace) -> int:
    """Train a bounded gross forecaster without creating execution authority."""

    from .microstructure_warehouse import MicrostructureWarehouse
    from .tape_depth_features import build_tape_depth_forecast_dataset
    from .tape_depth_model import (
        save_tape_depth_model_artifact,
        train_tape_depth_forecaster,
    )

    json_mode = bool(getattr(args, "json", False))
    symbol = str(getattr(args, "symbol", "BTCUSDT")).upper()
    horizon = int(getattr(args, "horizon_seconds", 60))
    latency = int(getattr(args, "total_latency_ms", 750))
    cadence = int(getattr(args, "decision_cadence_seconds", 5))
    window_days = int(getattr(args, "window_days", 180))
    output_path = Path(str(getattr(args, "output", "data/tape-depth-model.json")))
    if not 30 <= window_days <= 730:
        print("tape-depth-train: --window-days must lie in [30, 730]", file=sys.stderr)
        return 2

    def progress(phase: str, completed: int, total: int) -> None:
        if not json_mode:
            print(f"tape-depth-train {phase}: {completed}/{total}", flush=True)

    try:
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            latest_row = warehouse.connect().execute(
                "SELECT max(second_ms) FROM current_trade_1s WHERE symbol = ?",
                [symbol],
            ).fetchone()
            if latest_row is None or latest_row[0] is None:
                raise ValueError(f"no one-second trade rows exist for {symbol}")
            target_offset_seconds = max(1, int(math.ceil(latency / 1_000.0))) + horizon
            last_possible_decision_ms = (
                int(latest_row[0]) - target_offset_seconds * 1_000 + 1_000
            )
            end_value = getattr(args, "end_date", None)
            if end_value:
                end_date = _parse_cli_utc_date(end_value, "end-date")
                requested_end_ms = int(
                    datetime.combine(
                        end_date + timedelta(days=1),
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    ).timestamp()
                    * 1_000
                ) - 1
                end_ms = min(requested_end_ms, last_possible_decision_ms)
            else:
                end_ms = last_possible_decision_ms
            start_ms = end_ms - window_days * 86_400_000 + cadence * 1_000
            if not json_mode:
                print(
                    "tape-depth-train dataset: "
                    f"symbol={symbol} start_ms={start_ms} end_ms={end_ms} "
                    f"execution_claim=false",
                    flush=True,
                )
            dataset = build_tape_depth_forecast_dataset(
                warehouse,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                horizon_seconds=horizon,
                total_latency_ms=latency,
                decision_cadence_seconds=cadence,
                maximum_depth_age_ms=int(
                    getattr(args, "maximum_depth_age_ms", 60_000)
                ),
                maximum_rows=int(getattr(args, "maximum_rows", 5_000_000)),
                maximum_cached_rows=int(
                    getattr(args, "maximum_cached_rows", 15_000_000)
                ),
            )
        artifact = train_tape_depth_forecaster(
            dataset,
            risk_level=str(getattr(args, "risk_level", "conservative")),
            model_profile=str(getattr(args, "model_profile", "regularized")),
            feature_set=str(getattr(args, "feature_set", "full")),
            compute_backend=str(getattr(args, "compute_backend", "auto")),
            minimum_segment_rows=int(getattr(args, "minimum_segment_rows", 2_000)),
            progress=progress,
        )
        save_tape_depth_model_artifact(artifact, output_path)
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"tape-depth-train failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    payload = artifact.asdict()
    payload.pop("model_strings", None)
    payload["artifact_path"] = str(output_path)
    payload["artifact_sha256"] = digest
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        metrics = artifact.evaluation_metrics
        print(
            "tape-depth-train: "
            f"status={artifact.status} rows={metrics.rows} "
            f"risk={artifact.risk_level} profile={artifact.model_profile} "
            f"auc={metrics.direction_auc:.6f} "
            f"spearman_ic={metrics.spearman_information_coefficient:+.6f} "
            f"actions={metrics.calibration_threshold_rows} "
            f"long={metrics.calibration_threshold_long_rows} "
            f"short={metrics.calibration_threshold_short_rows} "
            "calibration_threshold_gross_bps="
            f"{metrics.calibration_threshold_signed_gross_bps:+.4f} "
            "trading_authority=false execution_claim=false"
        )
        if artifact.rejection_reasons:
            print("rejected: " + "; ".join(artifact.rejection_reasons), file=sys.stderr)
        print(f"artifact={output_path} sha256={digest}")
    return 0 if artifact.status == "research_candidate" else 2


def command_tape_depth_design(args: argparse.Namespace) -> int:
    """Persist the complete candidate set before any screening result exists."""

    from .model_experiment import tape_depth_candidate_design

    try:
        design = tape_depth_candidate_design(
            str(getattr(args, "risk_level", "conservative")),
            sampled_count=int(getattr(args, "sampled_count", 24)),
            seed=int(getattr(args, "seed", 20_260_710)),
        )
        output = Path(
            str(
                getattr(
                    args,
                    "output",
                    "data/tape-depth-experiment-design.json",
                )
            )
        )
        write_json_atomic(output, design.asdict(), indent=2, sort_keys=True)
    except (OSError, TypeError, ValueError) as exc:
        print(f"tape-depth-design failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    payload = design.asdict()
    payload["output"] = str(output)
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "tape-depth-design: "
            f"trials={design.trial_burden} anchors={design.anchor_count} "
            f"sampled={design.sampled_count} sha256={design.design_sha256}"
        )
        print(f"design={output}")
    return 0


def command_tape_depth_study(args: argparse.Namespace) -> int:
    """Run a complete design sequentially and freeze a screening winner."""

    from .assets import normalize_symbols
    from .microstructure_warehouse import MicrostructureWarehouse
    from .tape_depth_study import run_tape_depth_screening_study

    symbols = normalize_symbols(
        str(getattr(args, "symbols", "BTCUSDT,ETHUSDT,SOLUSDT")),
        default=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    )
    if any(not is_supported_major_symbol(symbol, "USDT") for symbol in symbols):
        print(
            "tape-depth-study supports only BTCUSDT, ETHUSDT, and SOLUSDT",
            file=sys.stderr,
        )
        return 2
    json_mode = bool(getattr(args, "json", False))

    def progress(phase: str, completed: int, total: int) -> None:
        if json_mode:
            return
        if phase == "candidate_started":
            print(
                f"tape-depth-study candidate {completed + 1}/{total} started",
                flush=True,
            )
        elif phase == "candidate_complete":
            print(
                f"tape-depth-study candidate {completed}/{total} complete",
                flush=True,
            )
        else:
            print(
                f"tape-depth-study {phase}: {completed}/{total}",
                flush=True,
            )

    options = {
        "symbols": symbols,
        "design_path": str(getattr(args, "design", "")),
        "output_dir": str(getattr(args, "output_dir", "data/tape-depth-study")),
        "training_window_days": int(getattr(args, "training_window_days", 730)),
        "tuning_window_days": int(getattr(args, "tuning_window_days", 30)),
        "calibration_window_days": int(
            getattr(args, "calibration_window_days", 30)
        ),
        "evaluation_window_days": int(
            getattr(args, "evaluation_window_days", 90)
        ),
        "total_latency_ms": int(getattr(args, "total_latency_ms", 750)),
        "maximum_rows": int(getattr(args, "maximum_rows", 5_000_000)),
        "maximum_cached_rows": int(
            getattr(args, "maximum_cached_rows", 15_000_000)
        ),
        "dataset_cache": bool(getattr(args, "dataset_cache", True)),
        "max_folds": int(getattr(args, "max_folds", 4)),
        "compute_backend": str(getattr(args, "compute_backend", "auto")),
        "minimum_segment_rows": int(
            getattr(args, "minimum_segment_rows", 10_000)
        ),
        "resume": bool(getattr(args, "resume", False)),
        "plan_only": bool(getattr(args, "plan_only", False)),
        "progress": progress,
    }
    try:
        if options["plan_only"]:
            report = run_tape_depth_screening_study(object(), **options)
        else:
            with MicrostructureWarehouse(
                str(getattr(args, "warehouse", "data/microstructure.duckdb")),
                cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
                memory_limit=str(getattr(args, "memory_limit", "8GB")),
                threads=int(getattr(args, "threads", 8)),
            ) as warehouse:
                report = run_tape_depth_screening_study(warehouse, **options)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"tape-depth-study failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if json_mode:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "tape-depth-study: "
            f"status={report['status']} "
            f"completed={report['completed_candidates']}/"
            f"{report['design']['trial_burden']} "
            "profitability_claim=false trading_authority=false"
        )
    if options["plan_only"]:
        return 0
    return 0 if report.get("status") == "winner_frozen" else 2


def command_tape_depth_prequential(args: argparse.Namespace) -> int:
    """Plan or run rolling, hash-bound gross-forecast evidence."""

    from .assets import normalize_symbols
    from .microstructure_warehouse import MicrostructureWarehouse
    from .tape_depth_prequential import run_tape_depth_prequential

    symbols = normalize_symbols(
        str(getattr(args, "symbols", "BTCUSDT,ETHUSDT,SOLUSDT")),
        default=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    )
    if any(not is_supported_major_symbol(symbol, "USDT") for symbol in symbols):
        print(
            "tape-depth-prequential supports only BTCUSDT, ETHUSDT, and SOLUSDT",
            file=sys.stderr,
        )
        return 2
    json_mode = bool(getattr(args, "json", False))
    horizon_arg = getattr(args, "horizon_seconds", None)
    cadence_arg = getattr(args, "decision_cadence_seconds", None)
    depth_age_arg = getattr(args, "maximum_depth_age_ms", None)

    def progress(phase: str, completed: int, total: int) -> None:
        if not json_mode:
            print(
                f"tape-depth-prequential {phase}: {completed}/{total}",
                flush=True,
            )

    try:
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            report = run_tape_depth_prequential(
                warehouse,
                symbols=symbols,
                output_dir=str(
                    getattr(args, "output_dir", "data/tape-depth-prequential")
                ),
                training_window_days=int(
                    getattr(args, "training_window_days", 730)
                ),
                tuning_window_days=int(getattr(args, "tuning_window_days", 30)),
                calibration_window_days=int(
                    getattr(args, "calibration_window_days", 30)
                ),
                evaluation_window_days=int(
                    getattr(args, "evaluation_window_days", 90)
                ),
                horizon_seconds=(None if horizon_arg is None else int(horizon_arg)),
                total_latency_ms=int(getattr(args, "total_latency_ms", 750)),
                decision_cadence_seconds=(
                    None if cadence_arg is None else int(cadence_arg)
                ),
                maximum_depth_age_ms=(
                    None if depth_age_arg is None else int(depth_age_arg)
                ),
                maximum_rows=int(getattr(args, "maximum_rows", 5_000_000)),
                maximum_cached_rows=int(
                    getattr(args, "maximum_cached_rows", 15_000_000)
                ),
                dataset_cache=bool(getattr(args, "dataset_cache", True)),
                study_stage=str(getattr(args, "study_stage", "development")),
                max_folds=int(getattr(args, "max_folds", 0)),
                risk_level=str(getattr(args, "risk_level", "conservative")),
                model_profile=getattr(args, "model_profile", None),
                feature_set=getattr(args, "feature_set", None),
                compute_backend=str(getattr(args, "compute_backend", "auto")),
                minimum_segment_rows=int(
                    getattr(args, "minimum_segment_rows", 10_000)
                ),
                selection_lock=getattr(args, "selection_lock", None),
                plan_only=bool(getattr(args, "plan_only", False)),
                resume=bool(getattr(args, "resume", False)),
                progress=progress,
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"tape-depth-prequential failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if json_mode:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif bool(getattr(args, "plan_only", False)):
        print(
            "tape-depth-prequential plan: "
            f"symbols={len(symbols)} folds={int(report['total_folds'])} "
            "trading_authority=false execution_claim=false"
        )
    else:
        aggregate = dict(report["aggregate_forecast_metrics"])
        print(
            "tape-depth-prequential: "
            f"status={report['status']} folds={report['completed_folds']} "
            f"rows={aggregate['rows']} "
            f"weighted_auc={float(aggregate['weighted_direction_auc']):.6f} "
            f"weighted_ic={float(aggregate['weighted_spearman_information_coefficient']):+.6f} "
            "profitability_claim=false trading_authority=false"
        )
    if bool(getattr(args, "plan_only", False)):
        return 0
    return 0 if report.get("status") == "research_candidate" else 2


def command_tape_depth_select(args: argparse.Namespace) -> int:
    """Freeze one winner without loading any terminal-fold report."""

    from .tape_depth_comparison import load_and_select_tape_depth_reports

    paths = tuple(str(path) for path in (getattr(args, "report", None) or ()))
    try:
        selection = load_and_select_tape_depth_reports(
            paths,
            output=str(getattr(args, "output", "data/tape-depth-selection.json")),
            design_path=str(getattr(args, "design", "")),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"tape-depth-select failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(selection, indent=2, sort_keys=True))
    else:
        print(
            "tape-depth-select: "
            f"status={selection['status']} "
            f"trials={selection['declared_trial_count']} "
            f"selected={selection['selected_trial']} "
            "profitability_claim=false trading_authority=false"
        )
    return 0 if selection.get("status") == "winner_frozen" else 2


def command_tape_depth_confirm(args: argparse.Namespace) -> int:
    """Evaluate the frozen winner on its untouched terminal folds."""

    from .tape_depth_comparison import load_and_confirm_tape_depth_report

    try:
        confirmation = load_and_confirm_tape_depth_report(
            selection_path=str(getattr(args, "selection", "")),
            report_path=str(getattr(args, "report", "")),
            output=str(
                getattr(args, "output", "data/tape-depth-confirmation.json")
            ),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"tape-depth-confirm failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if bool(getattr(args, "json", False)):
        print(json.dumps(confirmation, indent=2, sort_keys=True))
    else:
        print(
            "tape-depth-confirm: "
            f"status={confirmation['status']} "
            f"selected={confirmation['selected_trial']} "
            "profitability_claim=false trading_authority=false"
        )
    return 0 if confirmation.get("status") == "confirmed_forecast_candidate" else 2


def command_tape_depth_execution_confirm(args: argparse.Namespace) -> int:
    """Run the precommitted dates through real best-quote execution costs."""

    from .microstructure_warehouse import MicrostructureWarehouse
    from .tape_depth_execution_confirmation import (
        run_tape_depth_execution_confirmation,
    )

    json_mode = bool(getattr(args, "json", False))

    def progress(phase: str, completed: int, total: int) -> None:
        if not json_mode:
            print(
                f"tape-depth-execution-confirm {phase}: {completed}/{total}",
                flush=True,
            )

    try:
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            report = run_tape_depth_execution_confirmation(
                warehouse,
                design_path=str(getattr(args, "design", "")),
                availability_path=str(getattr(args, "availability", "")),
                output_dir=str(
                    getattr(
                        args,
                        "output_dir",
                        "data/tape-depth-execution-confirmation",
                    )
                ),
                resume=bool(getattr(args, "resume", False)),
                progress=progress,
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            "tape-depth-execution-confirm failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if json_mode:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        actual = dict(report["actual"])
        print(
            "tape-depth-execution-confirm: "
            f"status={report['status']} "
            f"periods={actual['completed_periods']} "
            f"executable={actual['combined_executable_rows']} "
            "mean_net_bps="
            f"{float(actual['combined_mean_net_return_bps']):+.6f} "
            "profitability_claim=false trading_authority=false"
        )
    return 0 if report.get("status") == "confirmed_after_cost_candidate" else 2


def command_fetch(args: argparse.Namespace) -> int:
    return data_workflows.command_fetch(
        args,
        load_runtime_fn=load_runtime,
        build_client_fn=_build_client,
    )


def _load_training_candles_from_db(
    db_path: str | Path,
    runtime: RuntimeConfig,
    *,
    interval: str,
    market_type: str,
    min_rows: int,
) -> list | None:
    return data_workflows.load_training_candles_from_db(
        db_path,
        runtime,
        interval=interval,
        market_type=market_type,
        min_rows=min_rows,
    )


def _confirm_download_missing_training_data(
    *,
    symbol: str,
    market_type: str,
    interval: str,
    available: int,
    required: int,
) -> bool:
    return data_workflows.confirm_download_missing_training_data(
        symbol=symbol,
        market_type=market_type,
        interval=interval,
        available=available,
        required=required,
        stdin=sys.stdin,
        input_fn=builtins.input,
    )


def _download_training_candles(args: argparse.Namespace, runtime: RuntimeConfig, *, interval: str, market_type: str) -> bool:
    return data_workflows.download_training_candles(
        args,
        runtime,
        interval=interval,
        market_type=market_type,
        command_fn=command_data_sync,
    )


def _load_training_candles(args: argparse.Namespace, runtime: RuntimeConfig) -> tuple[list | None, str]:
    return data_workflows.load_training_candles(
        args,
        runtime,
        load_rows_fn=_load_rows_for_command,
        db_loader_fn=_load_training_candles_from_db,
        confirm_fn=_confirm_download_missing_training_data,
        download_fn=_download_training_candles,
    )


def _classification_payload(report: object) -> dict[str, float | int]:
    return {
        "accuracy": float(getattr(report, "accuracy", 0.0)),
        "precision": float(getattr(report, "precision", 0.0)),
        "recall": float(getattr(report, "recall", 0.0)),
        "f1": float(getattr(report, "f1", 0.0)),
        "threshold": float(getattr(report, "threshold", 0.5)),
        "true_positive": int(getattr(report, "true_positive", 0)),
        "false_positive": int(getattr(report, "false_positive", 0)),
        "true_negative": int(getattr(report, "true_negative", 0)),
        "false_negative": int(getattr(report, "false_negative", 0)),
    }


def _threshold_classification_guard(baseline: object, candidate: object) -> dict[str, float | str | bool]:
    baseline_accuracy = float(getattr(baseline, "accuracy", 0.0))
    baseline_f1 = float(getattr(baseline, "f1", 0.0))
    baseline_precision = float(getattr(baseline, "precision", 0.0))
    candidate_accuracy = float(getattr(candidate, "accuracy", 0.0))
    candidate_f1 = float(getattr(candidate, "f1", 0.0))
    candidate_precision = float(getattr(candidate, "precision", 0.0))
    candidate_true_positive = int(getattr(candidate, "true_positive", 0))
    candidate_false_negative = int(getattr(candidate, "false_negative", 0))
    accuracy_floor = max(0.0, baseline_accuracy - 0.03)
    f1_floor = max(0.0, baseline_f1 - 0.05)
    precision_floor = max(0.0, baseline_precision - 0.02)
    detects_positive_cases = not (candidate_true_positive <= 0 < candidate_false_negative)
    stable_f1 = detects_positive_cases and candidate_accuracy >= accuracy_floor and candidate_f1 >= f1_floor
    conservative_precision = (
        detects_positive_cases
        and candidate_f1 > 0.0
        and candidate_accuracy >= baseline_accuracy + 0.02
        and candidate_precision >= precision_floor
    )
    passed = stable_f1 or conservative_precision
    if not detects_positive_cases:
        mode = "zero_true_positive"
    else:
        mode = "f1_stable" if stable_f1 else ("accuracy_precision" if conservative_precision else "rejected")
    return {
        "passed": bool(passed),
        "mode": mode,
        "accuracy_floor": float(accuracy_floor),
        "f1_floor": float(f1_floor),
        "precision_floor": float(precision_floor),
    }


def _threshold_capital_preservation_guard(
    profit_calibration: object,
    candidate_report: object | None = None,
) -> dict[str, float | int | str | bool]:
    baseline_score = float(getattr(profit_calibration, "baseline_score", 0.0))
    best_score = float(getattr(profit_calibration, "best_score", baseline_score))
    baseline_pnl = float(getattr(profit_calibration, "baseline_realized_pnl", 0.0))
    realized_pnl = float(getattr(profit_calibration, "realized_pnl", baseline_pnl))
    closed_trades = max(0, int(getattr(profit_calibration, "closed_trades", 0)))
    baseline_closed_trades = max(0, int(getattr(profit_calibration, "baseline_closed_trades", 0)))
    candidate_true_positive = int(getattr(candidate_report, "true_positive", 0)) if candidate_report else 1
    candidate_false_negative = int(getattr(candidate_report, "false_negative", 0)) if candidate_report else 0
    detects_positive_cases = not (candidate_true_positive <= 0 < candidate_false_negative)
    score_delta = best_score - baseline_score
    pnl_delta = realized_pnl - baseline_pnl
    material_pnl_improvement = pnl_delta >= max(1e-9, abs(baseline_pnl) * 0.10)
    tolerated_loss = -abs(baseline_pnl) * 0.10 if baseline_pnl < 0.0 else baseline_pnl
    passed = (
        bool(getattr(profit_calibration, "accepted", False))
        and detects_positive_cases
        and closed_trades > 0
        and score_delta >= 0.05
        and material_pnl_improvement
        and realized_pnl >= max(0.0, tolerated_loss)
    )
    return {
        "passed": bool(passed),
        "mode": "capital_preservation" if passed else ("zero_true_positive" if not detects_positive_cases else "rejected"),
        "score_delta": float(score_delta),
        "pnl_delta": float(pnl_delta),
        "closed_trades": int(closed_trades),
        "baseline_closed_trades": int(baseline_closed_trades),
        "tolerated_loss": float(tolerated_loss),
    }


def command_train(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    try:
        args = _apply_training_preset(args)
    except ValueError as exc:
        print(f"Training settings invalid: {exc}", file=sys.stderr)
        return 2
    cfg = load_strategy()
    runtime = load_runtime()
    candles, training_source = _load_training_candles(args, runtime)
    if candles is None:
        return 2
    try:
        compute_backend, _requested_backend_info = _workflow_compute_backend(
            runtime,
            getattr(args, "compute_backend", None),
            workflow="training",
        )
    except ValueError as exc:
        print(f"Training settings invalid: {exc}.", file=sys.stderr)
        return 2
    batch_size = max(1, int(getattr(args, "batch_size", 8192) or 8192))
    rows = _build_model_rows(candles, cfg, compute_backend=compute_backend)
    if not rows:
        print("No rows produced. Fetch more data or increase lookback.")
        return 2

    wf = None
    seed = int(getattr(args, "seed", 7))
    learning_rate = float(getattr(args, "learning_rate", 0.05))
    l2_penalty = float(getattr(args, "l2_penalty", 1e-4))
    if learning_rate <= 0.0:
        print("Training settings invalid: learning_rate must be > 0.", file=sys.stderr)
        return 2
    if l2_penalty < 0.0:
        print("Training settings invalid: l2_penalty must be >= 0.", file=sys.stderr)
        return 2
    if args.walk_forward:
        try:
            wf = walk_forward_report(
                rows,
                train_window=args.walk_forward_train,
                test_window=args.walk_forward_test,
                step=args.walk_forward_step,
                epochs=max(50, args.epochs // 2),
                calibrate=args.calibrate_threshold,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                compute_backend=compute_backend,
                batch_size=batch_size,
            )
            print(
                f"walk-forward: folds={wf['folds']} avg_score={wf['average_score']:.4f} "
                f"(train={wf['train_window']} test={wf['test_window']} step={wf['step']})"
            )
            fold_values = wf["scores"]
            if isinstance(fold_values, list) and fold_values:
                print(
                    "walk-forward fold scores: "
                    + ", ".join(f"{float(v):.3f}" for v in fold_values)
            )
        except ValueError as exc:
            print(f"walk-forward unavailable: {exc}")
            wf = None

    model_signature = _strategy_feature_signature(cfg)

    split = temporal_validation_split(rows)
    train_rows = split.train_rows
    calibration_rows = split.calibration_rows
    validation_rows = split.validation_rows

    model = train(
        train_rows,
        epochs=args.epochs,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
        seed=seed,
        feature_signature=model_signature,
        validation_rows=calibration_rows,
        early_stopping_rounds=max(5, min(25, int(args.epochs) // 5)) if calibration_rows else None,
        compute_backend=compute_backend,
        batch_size=batch_size,
    )
    backend_info = BackendInfo(
        requested=model.training_backend_requested,
        kind=cast(Any, model.training_backend_kind),
        device=model.training_backend_device,
        vendor=model.training_backend_vendor,
        reason=model.training_backend_reason,
    )
    probability_calibration = (
        calibrate_probability_temperature(
            calibration_rows,
            model,
            compute_backend=compute_backend,
            batch_size=batch_size,
        )
        if calibration_rows
        else assess_probability_calibration([], model)
    )
    if probability_calibration.status != "fail":
        model.probability_temperature = float(probability_calibration.temperature)
        model.probability_calibration_size = int(probability_calibration.rows)
        model.probability_log_loss_before = float(probability_calibration.log_loss_before)
        model.probability_log_loss_after = float(probability_calibration.log_loss_after)
        model.probability_brier_before = float(probability_calibration.brier_before)
        model.probability_brier_after = float(probability_calibration.brier_after)
        model.probability_ece_before = float(probability_calibration.expected_calibration_error_before)
        model.probability_ece_after = float(probability_calibration.expected_calibration_error_after)
        model.probability_calibration_backend_requested = str(probability_calibration.calibration_backend_requested)
        model.probability_calibration_backend_kind = str(probability_calibration.calibration_backend_kind)
        model.probability_calibration_backend_device = str(probability_calibration.calibration_backend_device)
        model.probability_calibration_backend_reason = str(probability_calibration.calibration_backend_reason)
    threshold = cfg.signal_threshold
    threshold_source = "strategy"
    threshold_calibration: dict[str, object] | None = None
    if args.calibrate_threshold and calibration_rows:
        strategy_report = evaluate_classification(calibration_rows, model, threshold=cfg.signal_threshold)
        classification_threshold = calibrate_threshold(calibration_rows, model, start=0.05, end=0.95, steps=31)
        classification_report = evaluate_classification(calibration_rows, model, threshold=classification_threshold)
        profit_calibration = calibrate_threshold_for_backtest(
            calibration_rows,
            model,
            cfg,
            starting_cash=1000.0,
            market_type=runtime.market_type,
            baseline_threshold=cfg.signal_threshold,
            start=0.05,
            end=0.95,
            steps=181,
            compute_backend=compute_backend,
            score_batch_size=batch_size,
        )
        profit_report = evaluate_classification(calibration_rows, model, threshold=profit_calibration.best_threshold)
        classification_guard = _threshold_classification_guard(strategy_report, profit_report)
        classification_guard_passed = bool(classification_guard["passed"])
        capital_guard = _threshold_capital_preservation_guard(profit_calibration, profit_report)
        capital_guard_passed = bool(capital_guard["passed"])
        if profit_calibration.accepted and (classification_guard_passed or capital_guard_passed):
            threshold = profit_calibration.threshold
            threshold_source = "profit_backtest"
        threshold_calibration = {
            "source": threshold_source,
            "strategy_baseline": _classification_payload(strategy_report),
            "classification": _classification_payload(classification_report),
            "profit_candidate": _classification_payload(profit_report),
            "profit_backtest": profit_calibration.asdict(),
            "classification_guard": classification_guard,
            "capital_guard": capital_guard,
        }
    model.decision_threshold = float(threshold)
    model.calibration_size = len(calibration_rows) if args.calibrate_threshold else 0
    model.validation_size = len(validation_rows)
    model.training_cutoff_timestamp = train_rows[-1].timestamp if train_rows else None
    model.threshold_source = threshold_source
    if threshold_calibration:
        profit_backtest = cast(dict[str, object], threshold_calibration["profit_backtest"])
        threshold_score = cast(float | int | str, profit_backtest["score"])
        threshold_pnl = cast(float | int | str, profit_backtest["realized_pnl"])
        threshold_trades = cast(float | int | str, profit_backtest["closed_trades"])
        model.threshold_calibration_score = float(threshold_score)
        model.threshold_calibration_pnl = float(threshold_pnl)
        model.threshold_calibration_trades = int(threshold_trades)
        model.threshold_diagnostic_best_threshold = float(
            cast(float | int | str, profit_backtest.get("best_threshold", threshold))
        )
        model.threshold_diagnostic_best_score = float(
            cast(float | int | str, profit_backtest.get("best_score", threshold_score))
        )
        model.threshold_diagnostic_best_pnl = float(
            cast(float | int | str, profit_backtest.get("best_realized_pnl", threshold_pnl))
        )
        model.threshold_diagnostic_best_trades = int(
            cast(float | int | str, profit_backtest.get("best_closed_trades", threshold_trades))
        )
    train_score = evaluate(train_rows, model, threshold=threshold)
    calibration_score = evaluate(calibration_rows, model, threshold=threshold) if calibration_rows else 0.0
    validation_score = evaluate(validation_rows, model, threshold=threshold) if validation_rows else 0.0
    quality = build_model_quality_report(train_rows, validation_rows, model, threshold)
    drift = feature_drift_report(validation_rows if validation_rows else rows[-min(50, len(rows)):], model)
    model.quality_score = float(quality.quality_score)
    model.quality_warnings = list(quality.warnings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialize_model(model, output)
    print(f"trained model saved to {output}")
    print(
        f"rows: {len(rows)} split train={len(train_rows)} "
        f"calibration={len(calibration_rows)} validation={len(validation_rows)}"
    )
    print(f"in-sample directional accuracy: {train_score:.3f}")
    print(f"out-of-sample directional accuracy: {validation_score:.3f}")
    if args.calibrate_threshold and calibration_rows:
        print(f"validated threshold: {threshold:.3f} source={threshold_source}")
        print(f"calibration directional accuracy: {calibration_score:.3f}")
        threshold_calibration_payload = cast(dict[str, object], threshold_calibration)
        profit_backtest = cast(dict[str, object], threshold_calibration_payload["profit_backtest"])
        accepted = "accepted" if threshold_source == "profit_backtest" else "rejected"
        best_threshold = float(cast(float | int | str, profit_backtest["best_threshold"]))
        best_score = float(cast(float | int | str, profit_backtest["best_score"]))
        baseline_score = float(cast(float | int | str, profit_backtest["baseline_score"]))
        print(
            "profit threshold candidate: "
            f"{accepted} best={best_threshold:.3f} "
            f"score={best_score:.3f} "
            f"baseline={baseline_score:.3f}"
        )
    print(f"model quality: {quality.status} score={quality.quality_score:.2f}")
    for warning in quality.warnings[:3]:
        print(f"model quality warning: {warning}")
    print(
        f"probability calibration: {probability_calibration.status} "
        f"temperature={probability_calibration.temperature:.2f} "
        f"brier={probability_calibration.brier_before:.3f}->{probability_calibration.brier_after:.3f} "
        f"log_loss={probability_calibration.log_loss_before:.3f}->{probability_calibration.log_loss_after:.3f}"
    )
    for warning in probability_calibration.warnings[:3]:
        print(f"probability calibration warning: {warning}")
    print(
        f"feature drift: {drift.status} max_z={drift.max_abs_z:.2f} "
        f"outliers={drift.outlier_fraction:.1%}"
    )
    for warning in drift.warnings[:3]:
        print(f"feature drift warning: {warning}")
    artifact = {
        "command": "train",
        "timestamp": int(time.time()),
        "seed": int(seed),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "train": {
            "input": str(args.input),
            "data_source": training_source,
            "db": str(getattr(args, "db", "data/market_data.sqlite")),
            "interval": str(getattr(args, "interval", None) or runtime.interval),
            "output": str(args.output),
            "rows_total": len(rows),
            "rows_train": len(train_rows),
            "rows_calibration": len(calibration_rows),
            "rows_validation": len(validation_rows),
            "epochs": int(args.epochs),
            "learning_rate": float(learning_rate),
            "l2_penalty": float(l2_penalty),
            "compute_backend": compute_backend,
            "batch_size": int(batch_size),
            "lookback_windows": list(cfg.feature_windows),
            "label_threshold": float(cfg.label_threshold),
            "preset": str(getattr(args, "requested_preset", args.preset)),
            "walk_forward": bool(args.walk_forward),
        },
        "walk_forward": wf if wf is not None else None,
        "metrics": {
            "in_sample_accuracy": float(train_score),
            "out_of_sample_accuracy": float(validation_score),
            "calibration_accuracy": float(calibration_score),
            "threshold": float(threshold),
            "tuned_threshold": float(threshold) if args.calibrate_threshold and calibration_rows else None,
            "threshold_source": threshold_source,
            "threshold_calibration": threshold_calibration,
            "calibrated_out_of_sample_accuracy": float(validation_score)
            if args.calibrate_threshold and validation_rows
            else None,
            "model_feature_version": model.feature_version,
            "model_feature_signature": model.feature_signature,
        },
        "model_quality": quality.asdict(),
        "probability_calibration": probability_calibration.asdict(),
        "feature_drift": drift.asdict(),
        "model": {
            "path": str(args.output),
            "feature_dim": int(model.feature_dim),
            "feature_version": str(model.feature_version),
            "feature_signature": model.feature_signature,
            "decision_threshold": float(model.decision_threshold)
            if model.decision_threshold is not None
            else None,
            "calibration_size": int(model.calibration_size),
            "validation_size": int(model.validation_size),
            "training_cutoff_timestamp": int(model.training_cutoff_timestamp)
            if model.training_cutoff_timestamp is not None
            else None,
            "best_epoch": int(model.best_epoch)
            if model.best_epoch is not None
            else None,
            "training_loss": float(model.training_loss)
            if model.training_loss is not None
            else None,
            "validation_loss": float(model.validation_loss)
            if model.validation_loss is not None
            else None,
            "quality_score": float(model.quality_score)
            if model.quality_score is not None
            else None,
            "quality_warnings": list(model.quality_warnings),
            "probability_temperature": float(model.probability_temperature),
            "probability_calibration_size": int(model.probability_calibration_size),
            "probability_log_loss_before": float(model.probability_log_loss_before)
            if model.probability_log_loss_before is not None
            else None,
            "probability_log_loss_after": float(model.probability_log_loss_after)
            if model.probability_log_loss_after is not None
            else None,
            "probability_brier_before": float(model.probability_brier_before)
            if model.probability_brier_before is not None
            else None,
            "probability_brier_after": float(model.probability_brier_after)
            if model.probability_brier_after is not None
            else None,
            "probability_ece_before": float(model.probability_ece_before)
            if model.probability_ece_before is not None
            else None,
            "probability_ece_after": float(model.probability_ece_after)
            if model.probability_ece_after is not None
            else None,
            "threshold_source": model.threshold_source,
            "threshold_calibration_score": float(model.threshold_calibration_score)
            if model.threshold_calibration_score is not None
            else None,
            "threshold_calibration_pnl": float(model.threshold_calibration_pnl)
            if model.threshold_calibration_pnl is not None
            else None,
            "threshold_calibration_trades": int(model.threshold_calibration_trades),
            "training_backend_requested": model.training_backend_requested,
            "training_backend_kind": model.training_backend_kind,
            "training_backend_device": model.training_backend_device,
            "training_backend_vendor": model.training_backend_vendor,
            "training_backend_reason": model.training_backend_reason,
        },
        "market": runtime.market_type,
        "symbol": runtime.symbol,
    }
    resolved_leverage = _resolve_futures_leverage(runtime, cfg)
    print(f"training backend: {describe_backend(backend_info)}")
    print(f"market={runtime.market_type} leverage={resolved_leverage:.2f}")
    artifact_path = _persist_run_artifact("train", Path(args.output).parent, artifact)
    print(f"saved train artifact to {artifact_path}")
    return 0


def command_tune(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    cfg = load_strategy()
    runtime = load_runtime()
    max_leverage = MAX_AUTONOMOUS_LEVERAGE
    if runtime.market_type == "futures" and runtime.api_key and runtime.api_secret:
        try:
            client = _build_client(runtime)
            max_leverage = min(MAX_AUTONOMOUS_LEVERAGE, float(client.get_max_leverage(runtime.symbol)))
        except BinanceAPIError:
            max_leverage = MAX_AUTONOMOUS_LEVERAGE
    candles = _load_rows_for_command(args.input, label="Tune data load failed")
    if candles is None:
        return 2
    try:
        candles = _filter_candles_for_time_window(
            candles,
            lookback_days=getattr(args, "lookback_days", None),
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
        )
    except ValueError as exc:
        print(f"Tune window invalid: {exc}", file=sys.stderr)
        return 2
    try:
        compute_backend, _backend_info = _workflow_compute_backend(
            runtime,
            getattr(args, "compute_backend", None),
            workflow="tuning",
        )
    except ValueError as exc:
        print(f"Tune settings invalid: {exc}.", file=sys.stderr)
        return 2
    batch_size = max(1, int(getattr(args, "batch_size", 8192) or 8192))
    rows = _build_model_rows(candles, cfg, compute_backend=compute_backend)
    if len(rows) < 40:
        print("Need more data rows to run tuning")
        return 2

    split = max(10, int(len(rows) * 0.7))
    train_rows = rows[:split]
    test_rows = rows[split:]

    risks: Iterable[float] = [args.min_risk + (args.max_risk - args.min_risk) * i / max(args.steps - 1, 1)
                              for i in range(args.steps)]
    levs: Iterable[float] = [args.min_leverage + (args.max_leverage - args.min_leverage) * i / max(args.steps - 1, 1)
                             for i in range(args.steps)]
    thrs: Iterable[float] = [args.min_threshold + (args.max_threshold - args.min_threshold) * i / max(args.steps - 1, 1)
                             for i in range(args.steps)]
    takes: Iterable[float] = [args.min_take + (args.max_take - args.min_take) * i / max(args.steps - 1, 1)
                              for i in range(args.steps)]
    stops: Iterable[float] = [args.min_stop + (args.max_stop - args.min_stop) * i / max(args.steps - 1, 1)
                              for i in range(args.steps)]
    best: StrategyConfig = cfg
    fallback: StrategyConfig | None = None
    fallback_score = float("-inf")
    tuned = False
    best_score = float("-inf")

    for risk in risks:
        for leverage in levs:
            for threshold in thrs:
                for take in takes:
                    for stop in stops:
                        candidate = StrategyConfig(
                            **{
                                **cfg.asdict(),
                                "risk_per_trade": max(0.0001, risk),
                                "leverage": max(1.0, min(float(max_leverage), leverage)),
                                "signal_threshold": max(0.05, min(0.95, threshold)),
                                "take_profit_pct": max(0.0, min(0.99, take)),
                                "stop_loss_pct": max(0.0, min(0.99, stop)),
                            },
                        )
                        model = train(
                            train_rows,
                            epochs=max(50, candidate.training_epochs // 2),
                            compute_backend=compute_backend,
                            batch_size=batch_size,
                        )
                        candidate_result = run_backtest(
                            test_rows,
                            model,
                            candidate,
                            market_type=runtime.market_type,
                            starting_cash=1000.0,
                            compute_backend=compute_backend,
                            score_batch_size=batch_size,
                        )
                        score = _tune_score(candidate_result, starting_cash=1000.0)
                        candidate_stopped = bool(getattr(candidate_result, "stopped_by_drawdown", False))
                        if candidate_stopped:
                            if score > fallback_score:
                                fallback_score = score
                                fallback = candidate
                            continue
                        if score > best_score:
                            best_score = score
                            best = candidate
                            tuned = True
        # no valid candidate should silently keep -inf
    if not tuned:
        if fallback is not None:
            best = fallback
            best_score = fallback_score
            print("Warning: all tune candidates hit drawdown limit; using best fallback score by risk-adjusted metric.")
        else:
            print("No valid candidates evaluated.")
            return 2

    print(f"tune best score: {best_score:.4f}")
    print(
        f"tune best config risk={best.risk_per_trade:.4f} take={best.take_profit_pct:.4f} "
        f"stop={best.stop_loss_pct:.4f} leverage={best.leverage:.1f} threshold={best.signal_threshold:.3f}"
    )
    if args.save_best:
        save_strategy(best)
        print("Saved tuned strategy.")
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    cfg = load_strategy()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 2

    candles = _load_rows_for_command(args.input, label="Backtest data load failed")
    if candles is None:
        return 2
    try:
        model = _load_readiness_model(model_path, cfg)[0]
    except (OSError, json.JSONDecodeError, ModelLoadError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    try:
        compute_backend, _backend_info = _workflow_compute_backend(
            runtime,
            getattr(args, "compute_backend", None),
            workflow="backtest scoring",
        )
    except ValueError as exc:
        print(f"Backtest settings invalid: {exc}.", file=sys.stderr)
        return 2
    cfg = apply_model_strategy_overrides(cfg, model)
    rows = _backtest_rows_for_model(candles, cfg, model, compute_backend=compute_backend)
    data_coverage = describe_candle_coverage(
        symbol=runtime.symbol,
        market_type=runtime.market_type,
        interval=runtime.interval,
        available_candles=candles,
        used_candles=candles,
        rows_used=len(rows),
        source_scope="json_file_loaded_candles",
    )
    decision_threshold = model_decision_threshold(model, cfg.signal_threshold)
    score_batch_size = max(1, int(getattr(args, "score_batch_size", 8192) or 8192))
    execution_profile = _load_cli_execution_profile(args, runtime, cfg, workflow="backtest execution profile")
    result = run_backtest(
        rows,
        model,
        cfg,
        starting_cash=args.start_cash,
        market_type=runtime.market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        symbol_profile=execution_profile.profile,
    )
    artifact = {
        "command": "backtest",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "input": str(args.input),
        "model": str(model_path),
        "decision_threshold": float(decision_threshold),
        "starting_cash": float(args.start_cash),
        "rows": len(rows),
        "data_coverage": data_coverage.asdict(),
        "market": runtime.market_type,
        "symbol": runtime.symbol,
        "scoring_backend": {
            "requested": result.scoring_backend_requested,
            "kind": result.scoring_backend_kind,
            "device": result.scoring_backend_device,
            "reason": result.scoring_backend_reason,
            "score_batch_size": score_batch_size,
        },
        "execution_profile": execution_profile.asdict(),
        "result": {
            "trades": int(result.trades),
            "win_rate": float(result.win_rate),
            "realized_pnl": float(result.realized_pnl),
            "fees": float(result.total_fees),
            "max_exposure": float(result.max_exposure),
            "ending_cash": float(result.ending_cash),
            "max_drawdown": float(result.max_drawdown),
            "stopped_by_drawdown": bool(result.stopped_by_drawdown),
            "trades_per_day_cap_hit": int(result.trades_per_day_cap_hit),
            "closed_trades": int(result.closed_trades),
            "gross_exposure": float(result.gross_exposure),
            "buy_hold_pnl": float(result.buy_hold_pnl),
            "edge_vs_buy_hold": float(result.edge_vs_buy_hold),
            "equity_curve_points": len(getattr(result, "equity_curve", ()) or ()),
            "trade_return_count": len(getattr(result, "trade_returns", ()) or ()),
            "trade_pnls": list(getattr(result, "trade_pnls", ()) or ()),
            "trade_returns": list(getattr(result, "trade_returns", ()) or ()),
            "gross_profit": float(getattr(result, "gross_profit", 0.0)),
            "gross_loss": float(getattr(result, "gross_loss", 0.0)),
            "profit_factor": float(getattr(result, "profit_factor", 0.0)),
            "expectancy": float(getattr(result, "expectancy", 0.0)),
            "average_trade_return": float(getattr(result, "average_trade_return", 0.0)),
            "trade_return_stdev": float(getattr(result, "trade_return_stdev", 0.0)),
            "max_consecutive_losses": int(getattr(result, "max_consecutive_losses", 0)),
        },
    }
    _persist_run_artifact("backtest", model_path.parent, artifact)

    print(f"day-trading backtest {runtime.symbol}")
    print(f"market: {runtime.market_type}")
    print(
        "data_span: "
        f"{data_coverage.used_start_utc or 'n/a'} -> {data_coverage.used_end_utc or 'n/a'} "
        f"interval={data_coverage.interval} candles={data_coverage.candles_used} "
        f"rows={data_coverage.rows_used} years={data_coverage.used_duration_years:.2f} "
        f"coverage={data_coverage.coverage_ratio:.4f} gaps={data_coverage.gap_count}"
    )
    print(f"scoring_backend: {result.scoring_backend_kind} device={result.scoring_backend_device}")
    if result.scoring_backend_reason:
        print(f"scoring_backend_reason: {result.scoring_backend_reason}")
    _print_execution_profile_evidence(execution_profile)
    print(f"trades: {result.trades}")
    print(f"win_rate: {result.win_rate:.2%}")
    print(f"realized_pnl: {result.realized_pnl:.2f}")
    print(f"fees: {result.total_fees:.2f}")
    print(f"max_exposure: {result.max_exposure:.2f}")
    print(f"starting_cash: {result.starting_cash:.2f}")
    print(f"ending_cash: {result.ending_cash:.2f}")
    print(f"buy_hold_pnl: {result.buy_hold_pnl:.2f}")
    print(f"edge_vs_buy_hold: {result.edge_vs_buy_hold:.2f}")
    print(f"profit_factor: {float(getattr(result, 'profit_factor', 0.0)):.2f}")
    print(f"expectancy: {float(getattr(result, 'expectancy', 0.0)):.2f}")
    print(f"average_trade_return: {float(getattr(result, 'average_trade_return', 0.0)):.2%}")
    print(f"max_consecutive_losses: {int(getattr(result, 'max_consecutive_losses', 0))}")
    print(f"max_drawdown: {result.max_drawdown:.2%}")
    print(f"stopped_by_drawdown: {result.stopped_by_drawdown}")
    return 0


def command_microstructure_prequential(args: argparse.Namespace) -> int:
    from .microstructure_features import (
        apply_path_aware_lifecycle_targets,
        build_executable_microstructure_dataset,
    )
    from .microstructure_model import load_microstructure_model_artifact
    from .microstructure_prequential import (
        PrequentialConfig,
        evaluate_prequential_microstructure_model,
    )
    from .microstructure_warehouse import MicrostructureWarehouse

    json_mode = bool(getattr(args, "json", False))

    def model_progress(phase: str, completed: int, total: int) -> None:
        if not json_mode:
            print(
                f"microstructure-prequential {phase}: {completed}/{total}",
                flush=True,
            )

    try:
        artifact = load_microstructure_model_artifact(
            Path(str(getattr(args, "input", "data/microstructure-model.json")))
        )
        if artifact.status != "candidate" or artifact.rejection_reasons:
            raise ValueError(
                "microstructure-prequential requires an unrejected candidate artifact"
            )
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            warehouse.rebuild_causal_feature_bars(artifact.symbol)
            dataset = build_executable_microstructure_dataset(
                warehouse,
                symbol=artifact.symbol,
                horizon_seconds=artifact.horizon_seconds,
                total_latency_ms=artifact.total_latency_ms,
                taker_fee_bps=artifact.taker_fee_bps,
                additional_slippage_bps_per_side=(
                    artifact.additional_slippage_bps_per_side
                ),
                max_quote_age_ms=artifact.max_quote_age_ms,
                reference_order_notional_quote=(
                    artifact.reference_order_notional_quote
                ),
                max_l1_participation=artifact.max_l1_participation,
                decision_cadence_seconds=artifact.decision_cadence_seconds,
            )
            if artifact.stop_loss_bps is not None or artifact.take_profit_bps is not None:
                if (
                    artifact.stop_loss_bps is None
                    or artifact.take_profit_bps is None
                    or artifact.trigger_execution_slippage_bps is None
                ):
                    raise ValueError("candidate path-aware target contract is incomplete")
                dataset, _path_evidence = apply_path_aware_lifecycle_targets(
                    warehouse,
                    dataset,
                    stop_loss_bps=artifact.stop_loss_bps,
                    take_profit_bps=artifact.take_profit_bps,
                    trigger_execution_slippage_bps=(
                        artifact.trigger_execution_slippage_bps
                    ),
                )
            report = evaluate_prequential_microstructure_model(
                artifact,
                dataset,
                config=PrequentialConfig(
                    training_window_days=int(
                        getattr(args, "training_window_days", 180)
                    ),
                    minimum_training_days=int(
                        getattr(args, "minimum_training_days", 60)
                    ),
                    calibration_days=int(getattr(args, "calibration_days", 14)),
                    policy_days=int(getattr(args, "policy_days", 14)),
                    evaluation_block_days=int(
                        getattr(args, "evaluation_block_days", 7)
                    ),
                    minimum_segment_rows=int(
                        getattr(args, "minimum_segment_rows", 256)
                    ),
                    minimum_class_rows=int(
                        getattr(args, "minimum_class_rows", 128)
                    ),
                    bootstrap_samples=int(
                        getattr(args, "bootstrap_samples", 2000)
                    ),
                    max_folds=int(getattr(args, "max_folds", 0)),
                ),
                compute_backend=str(getattr(args, "compute_backend", "auto")),
                predictions_path=Path(
                    str(
                        getattr(
                            args,
                            "predictions",
                            "data/microstructure-prequential-predictions.csv",
                        )
                    )
                ),
                chart_path=Path(
                    str(
                        getattr(
                            args,
                            "chart",
                            "data/microstructure-prequential.svg",
                        )
                    )
                ),
                report_path=Path(
                    str(
                        getattr(
                            args,
                            "output",
                            "data/microstructure-prequential.json",
                        )
                    )
                ),
                progress=model_progress,
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"microstructure-prequential failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    payload = report.asdict()
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        metrics = report.aggregate["metrics"]
        confidence = report.aggregate["confidence"]
        print(
            "microstructure-prequential: "
            f"status={report.status} folds={report.coverage['complete_folds']}/"
            f"{report.coverage['planned_folds']} rows={report.coverage['evaluated_rows']} "
            f"trades={metrics['trades']} net_bps={float(metrics['total_net_bps']):+.4f} "
            f"max_dd_bps={float(metrics['max_drawdown_bps']):.4f} "
            f"daily_ci95=[{float(confidence['mean_daily_net_bps_ci_lower']):+.4f},"
            f"{float(confidence['mean_daily_net_bps_ci_upper']):+.4f}]"
        )
        if report.reasons:
            print(f"  first_rejection={report.reasons[0]}")
        print(f"  predictions -> {report.predictions_path}")
        print(f"  chart -> {report.chart_path}")
        print(f"  report -> {getattr(args, 'output', 'data/microstructure-prequential.json')}")
        print("  terminal_holdout=not_accessed trading_authority=false")
    return 0 if report.passed else 2


def command_microstructure_promote(args: argparse.Namespace) -> int:
    from .microstructure_features import (
        apply_path_aware_lifecycle_targets,
        build_executable_microstructure_dataset,
    )
    from .microstructure_model import (
        evaluate_microstructure_model_terminal,
        load_microstructure_model_artifact,
        microstructure_candidate_sha256,
        refit_validated_microstructure_model,
        save_microstructure_model_artifact,
    )
    from .microstructure_prequential import attach_verified_prequential_evidence
    from .microstructure_warehouse import MicrostructureWarehouse

    input_path = Path(str(getattr(args, "input", "data/microstructure-model.json")))
    output_value = getattr(args, "output", None)
    output_path = Path(str(output_value)) if output_value else input_path
    json_mode = bool(getattr(args, "json", False))
    terminal_reservation: dict[str, object] | None = None
    deployment_refit_error: str | None = None

    def progress(phase: str, completed: int, total: int) -> None:
        if not json_mode:
            print(f"microstructure-promote {phase}: {completed}/{total}", flush=True)

    try:
        artifact = load_microstructure_model_artifact(input_path)
        if artifact.status != "candidate" or artifact.rejection_reasons:
            raise ValueError(
                "microstructure-promote requires an unrejected candidate artifact"
            )
        if artifact.prequential_validation is not None:
            raise ValueError(
                "microstructure-promote requires the original unattached candidate"
            )
        with MicrostructureWarehouse(
            str(getattr(args, "warehouse", "data/microstructure.duckdb")),
            cache_root=str(getattr(args, "cache_root", "data/archive-cache")),
            memory_limit=str(getattr(args, "memory_limit", "8GB")),
            threads=int(getattr(args, "threads", 8)),
        ) as warehouse:
            warehouse.require_causal_feature_bars(artifact.symbol)
            dataset = build_executable_microstructure_dataset(
                warehouse,
                symbol=artifact.symbol,
                horizon_seconds=artifact.horizon_seconds,
                total_latency_ms=artifact.total_latency_ms,
                taker_fee_bps=artifact.taker_fee_bps,
                additional_slippage_bps_per_side=(
                    artifact.additional_slippage_bps_per_side
                ),
                max_quote_age_ms=artifact.max_quote_age_ms,
                reference_order_notional_quote=(
                    artifact.reference_order_notional_quote
                ),
                max_l1_participation=artifact.max_l1_participation,
                decision_cadence_seconds=artifact.decision_cadence_seconds,
            )
            if (
                artifact.stop_loss_bps is None
                or artifact.take_profit_bps is None
                or artifact.trigger_execution_slippage_bps is None
            ):
                raise ValueError(
                    "microstructure-promote requires a path-aware protective-exit candidate"
                )
            dataset, _path_evidence = apply_path_aware_lifecycle_targets(
                warehouse,
                dataset,
                stop_loss_bps=artifact.stop_loss_bps,
                take_profit_bps=artifact.take_profit_bps,
                trigger_execution_slippage_bps=(
                    artifact.trigger_execution_slippage_bps
                ),
            )
            artifact = attach_verified_prequential_evidence(
                artifact,
                dataset,
                report_path=Path(
                    str(
                        getattr(
                            args,
                            "prequential_report",
                            "data/microstructure-prequential.json",
                        )
                    )
                ),
                predictions_path=Path(
                    str(
                        getattr(
                            args,
                            "prequential_predictions",
                            "data/microstructure-prequential-predictions.csv",
                        )
                    )
                ),
                chart_path=Path(
                    str(
                        getattr(
                            args,
                            "prequential_chart",
                            "data/microstructure-prequential.svg",
                        )
                    )
                ),
            )
            save_microstructure_model_artifact(artifact, output_path)
            source_evidence = dict(dataset.source_evidence or {})
            reservation = warehouse.reserve_terminal_holdout(
                symbol=dataset.symbol,
                first_utc_day=artifact.split.terminal_start_ms // 86_400_000,
                last_utc_day=int(dataset.decision_time_ms[-1]) // 86_400_000,
                candidate_sha256=microstructure_candidate_sha256(artifact),
                source_manifest_fingerprint=str(
                    source_evidence["manifest_fingerprint"]
                ),
                source_feature_build_id=str(source_evidence["build_id"]),
                feature_version=artifact.feature_version,
                model_schema_version=artifact.schema_version,
                prequential_report_sha256=(
                    artifact.prequential_validation.report_sha256
                    if artifact.prequential_validation is not None
                    else ""
                ),
            )
            terminal_reservation = dict(reservation)
            try:
                artifact = evaluate_microstructure_model_terminal(
                    artifact,
                    dataset,
                    compute_backend=str(
                        getattr(args, "compute_backend", "auto")
                    ),
                    progress=progress,
                )
                save_microstructure_model_artifact(artifact, output_path)
            except Exception as exc:
                terminal_reservation.update(
                    warehouse.finalize_terminal_holdout(
                        str(reservation["reservation_id"]),
                        result_status="evaluation_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
            terminal_reservation.update(
                warehouse.finalize_terminal_holdout(
                    str(reservation["reservation_id"]),
                    result_status=artifact.status,
                )
            )
            if artifact.status == "validated":
                try:
                    artifact = refit_validated_microstructure_model(
                        artifact,
                        dataset,
                        compute_backend=str(
                            getattr(args, "compute_backend", "auto")
                        ),
                        progress=progress,
                    )
                    save_microstructure_model_artifact(artifact, output_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    deployment_refit_error = f"{type(exc).__name__}: {exc}"
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"microstructure-promote failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    digest = save_microstructure_model_artifact(artifact, output_path)
    payload = artifact.asdict()
    payload.pop("model_strings", None)
    payload.pop("deployment_model_strings", None)
    payload["artifact_path"] = str(output_path)
    payload["artifact_sha256"] = digest
    payload["terminal_holdout_reservation"] = terminal_reservation
    payload["deployment_refit_error"] = deployment_refit_error
    payload["trading_authority"] = artifact.status == "accepted"
    payload["shadow_required"] = artifact.status == "shadow_candidate"
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        terminal_metrics = artifact.terminal_metrics
        print(
            "microstructure-promote: "
            f"status={artifact.status} "
            f"terminal_trades={terminal_metrics.trades if terminal_metrics else 0} "
            f"terminal_net_bps="
            f"{terminal_metrics.total_net_bps if terminal_metrics else 0.0:+.4f} "
            f"trading_authority={str(artifact.status == 'accepted').lower()} "
            f"shadow_required={str(artifact.status == 'shadow_candidate').lower()}"
        )
        if artifact.rejection_reasons:
            print("rejected: " + "; ".join(artifact.rejection_reasons), file=sys.stderr)
        if deployment_refit_error:
            print("deployment refit failed: " + deployment_refit_error, file=sys.stderr)
        print(f"artifact={output_path} sha256={digest}")
    return (
        0
        if artifact.status == "shadow_candidate" and deployment_refit_error is None
        else 2
    )


def command_backtest_chart(args: argparse.Namespace) -> int:
    from .performance_charts import EquityPoint, write_equity_svg

    runtime = load_runtime()
    cfg = load_strategy()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 2
    candles = _load_rows_for_command(args.input, label="Backtest chart data load failed")
    if candles is None:
        return 2
    try:
        model = _load_readiness_model(model_path, cfg)[0]
    except (OSError, json.JSONDecodeError, ModelLoadError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    try:
        compute_backend, _backend_info = _workflow_compute_backend(
            runtime,
            getattr(args, "compute_backend", None),
            workflow="backtest chart scoring",
        )
    except ValueError as exc:
        print(f"Backtest chart settings invalid: {exc}.", file=sys.stderr)
        return 2
    cfg = apply_model_strategy_overrides(cfg, model)
    rows = _backtest_rows_for_model(candles, cfg, model, compute_backend=compute_backend)
    data_coverage = describe_candle_coverage(
        symbol=runtime.symbol,
        market_type=runtime.market_type,
        interval=runtime.interval,
        available_candles=candles,
        used_candles=candles,
        rows_used=len(rows),
        source_scope="json_file_loaded_candles",
    )
    if not rows:
        print("Backtest chart failed: no rows available.", file=sys.stderr)
        return 2
    execution_profile = _load_cli_execution_profile(args, runtime, cfg, workflow="backtest chart execution profile")
    result = run_backtest(
        rows,
        model,
        cfg,
        starting_cash=float(args.start_cash),
        market_type=runtime.market_type,
        compute_backend=compute_backend,
        score_batch_size=max(1, int(getattr(args, "score_batch_size", 8192))),
        symbol_profile=execution_profile.profile,
    )
    points = [
        EquityPoint(
            int(index),
            float(point["equity"]),
            float(point["drawdown"]),
            int(point["timestamp"]) if "timestamp" in point else None,
        )
        for index, point in enumerate(getattr(result, "equity_curve", ()) or ())
        if isinstance(point, dict) and "equity" in point and "drawdown" in point
    ]
    if not points:
        points = [
            EquityPoint(0, float(result.starting_cash), 0.0),
            EquityPoint(1, float(result.ending_cash), float(result.max_drawdown)),
        ]
    output = write_equity_svg(points, args.output, title=f"{runtime.symbol} day-trading backtest")
    print(f"backtest chart saved to {output}")
    print(
        "data_span: "
        f"{data_coverage.used_start_utc or 'n/a'} -> {data_coverage.used_end_utc or 'n/a'} "
        f"interval={data_coverage.interval} candles={data_coverage.candles_used} "
        f"rows={data_coverage.rows_used} years={data_coverage.used_duration_years:.2f} "
        f"coverage={data_coverage.coverage_ratio:.4f} gaps={data_coverage.gap_count}"
    )
    _print_execution_profile_evidence(execution_profile)
    print(
        f"ending_cash={result.ending_cash:.2f} realized_pnl={result.realized_pnl:.2f} "
        f"max_drawdown={result.max_drawdown:.2%}"
    )
    return 0


def _tune_score(result: object, starting_cash: float = 1000.0) -> float:
    return risk_adjusted_backtest_score(result, starting_cash=starting_cash)


def command_evaluate(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    cfg = load_strategy()
    runtime = load_runtime()
    candles = _load_rows_for_command(args.input, label="Evaluation data load failed")
    if candles is None:
        return 2
    base_rows = _build_model_rows(candles, cfg, compute_backend=runtime.compute_backend)
    if not base_rows:
        print("No rows available for evaluation. Fetch more data first.")
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 2

    try:
        model = _load_readiness_model(model_path, cfg)[0]
    except (OSError, json.JSONDecodeError, ModelLoadError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    cfg = apply_model_strategy_overrides(cfg, model)
    rows = _readiness_model_rows(candles, cfg, model, compute_backend=runtime.compute_backend)
    if not rows:
        print("No rows available for evaluation. Fetch more data first.")
        return 2

    split = temporal_validation_split(rows)
    train_rows = split.train_rows
    calibration_rows = split.calibration_rows
    test_rows = split.validation_rows

    threshold = model_decision_threshold(model, cfg.signal_threshold)
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif test_rows:
        # make default threshold robust against short samples and class imbalance
        threshold = model_decision_threshold(model, cfg.signal_threshold)

    if args.calibrate_threshold and calibration_rows:
        threshold = calibrate_threshold(calibration_rows, model, start=0.05, end=0.95, steps=31)

    report = evaluate_classification(test_rows if test_rows else rows, model, threshold=threshold)
    train_report = evaluate_classification(train_rows, model, threshold=threshold) if train_rows else None
    quality = build_model_quality_report(train_rows, test_rows if test_rows else rows, model, threshold)
    probability_calibration = assess_probability_calibration(test_rows if test_rows else rows, model)
    drift = feature_drift_report(test_rows if test_rows else rows, model)
    artifact = {
        "command": "evaluate",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "input": str(args.input),
        "model": str(args.model),
        "market": runtime.market_type,
        "symbol": runtime.symbol,
        "split": {
            "train_rows": len(train_rows),
            "calibration_rows": len(calibration_rows),
            "test_rows": len(test_rows),
        },
        "threshold": float(report.threshold),
        "calibrated": bool(args.calibrate_threshold),
        "rows": {
            "train": {
                "accuracy": float(train_report.accuracy) if train_report is not None else 0.0,
                "precision": float(train_report.precision) if train_report is not None else 0.0,
                "recall": float(train_report.recall) if train_report is not None else 0.0,
                "f1": float(train_report.f1) if train_report is not None else 0.0,
                "true_positive": int(train_report.true_positive) if train_report is not None else 0,
                "false_positive": int(train_report.false_positive) if train_report is not None else 0,
                "true_negative": int(train_report.true_negative) if train_report is not None else 0,
                "false_negative": int(train_report.false_negative) if train_report is not None else 0,
            },
            "test": {
                "accuracy": float(report.accuracy),
                "precision": float(report.precision),
                "recall": float(report.recall),
                "f1": float(report.f1),
                "true_positive": int(report.true_positive),
                "false_positive": int(report.false_positive),
                "true_negative": int(report.true_negative),
                "false_negative": int(report.false_negative),
            },
        },
        "model_quality": quality.asdict(),
        "probability_calibration": probability_calibration.asdict(),
        "feature_drift": drift.asdict(),
    }
    _persist_run_artifact("evaluate", Path(args.model).parent, artifact)

    print(f"evaluate model={model_path}")
    print(f"threshold: {report.threshold:.3f}")
    train_accuracy = train_report.accuracy if train_report is not None else 0.0
    train_precision = train_report.precision if train_report is not None else 0.0
    train_recall = train_report.recall if train_report is not None else 0.0
    train_f1 = train_report.f1 if train_report is not None else 0.0
    print(
        "train_accuracy: "
        f"{train_accuracy:.3f} precision={train_precision:.3f} "
        f"recall={train_recall:.3f} f1={train_f1:.3f}"
    )
    print(
        "test_accuracy: "
        f"{report.accuracy:.3f} precision={report.precision:.3f} "
        f"recall={report.recall:.3f} f1={report.f1:.3f}"
    )
    print(
        "confusion tp={tp} fp={fp} tn={tn} fn={fn}".format(
            tp=report.true_positive, fp=report.false_positive, tn=report.true_negative, fn=report.false_negative
        )
    )
    print(f"model_quality: {quality.status} score={quality.quality_score:.2f}")
    for warning in quality.warnings[:3]:
        print(f"quality_warning: {warning}")
    print(
        f"probability_calibration: {probability_calibration.status} "
        f"temperature={probability_calibration.temperature:.2f} "
        f"brier={probability_calibration.brier_after:.3f} "
        f"ece={probability_calibration.expected_calibration_error_after:.3f}"
    )
    for warning in probability_calibration.warnings[:3]:
        print(f"probability_warning: {warning}")
    print(
        f"feature_drift: {drift.status} max_z={drift.max_abs_z:.2f} "
        f"outliers={drift.outlier_fraction:.1%}"
    )
    for warning in drift.warnings[:3]:
        print(f"drift_warning: {warning}")
    return 0


def _external_signal_cache_path(model_path: Path) -> Path:
    return model_path.parent / "signals" / "external_signals.json"


def _strategy_with_external_risk(cfg: StrategyConfig, risk_multiplier: float) -> StrategyConfig:
    multiplier = _clamp(float(risk_multiplier), 0.0, 1.0)
    if multiplier >= 0.999:
        return cfg
    return StrategyConfig(
        **{
            **cfg.asdict(),
            "risk_per_trade": max(0.0001, cfg.risk_per_trade * multiplier),
            "max_position_pct": max(0.0, cfg.max_position_pct * multiplier),
        }
    )


def _strategy_with_size_multiplier(cfg: StrategyConfig, size_multiplier: float) -> StrategyConfig:
    multiplier = _clamp(float(size_multiplier), 0.0, 1.0)
    if multiplier >= 0.999:
        return cfg
    return StrategyConfig(
        **{
            **cfg.asdict(),
            "risk_per_trade": max(0.0001, cfg.risk_per_trade * multiplier),
            "max_position_pct": max(0.0, cfg.max_position_pct * multiplier),
        }
    )


def _apply_external_signal_to_score(
    score: float,
    cfg: StrategyConfig,
    report: ExternalSignalReport,
) -> tuple[float, StrategyConfig, float]:
    applied_adjustment = float(report.score_adjustment)
    if report.fresh_count < max(0, int(cfg.external_signal_min_providers)):
        applied_adjustment = min(0.0, applied_adjustment)
    max_adjustment = _clamp(float(cfg.external_signal_max_adjustment), 0.0, 0.20)
    applied_adjustment = _clamp(applied_adjustment, -max_adjustment, max_adjustment)
    adjusted_score = _clamp(float(score) + applied_adjustment, 0.0, 1.0)
    effective_cfg = _strategy_with_external_risk(cfg, report.risk_multiplier)
    return adjusted_score, effective_cfg, applied_adjustment


def _record_model_telemetry(
    cfg: StrategyConfig,
    *,
    step: int,
    row: ModelRow,
    raw_score: float,
    adjusted_score: float,
    threshold: float,
    model: object,
    runtime: RuntimeConfig,
) -> None:
    if not cfg.telemetry_enabled:
        return
    try:
        from .telemetry_store import TradingTelemetryStore

        payload = {
            "step": int(step),
            "timestamp": int(row.timestamp),
            "close": float(row.close),
            "features": [float(value) for value in row.features],
            "raw_score": float(raw_score),
            "adjusted_score": float(adjusted_score),
            "threshold": float(threshold),
            "model_type": model.__class__.__name__,
            "model_signature": str(getattr(model, "feature_signature", "") or ""),
            "training_backend_kind": str(getattr(model, "training_backend_kind", "") or ""),
            "runtime_compute_backend": str(runtime.compute_backend),
        }
        with TradingTelemetryStore(cfg.telemetry_db_path) as store:
            store.record_observation(
                kind="model_decision",
                source="internal_model",
                payload=payload,
                observed_at_ms=int(row.timestamp),
                symbol=runtime.symbol,
                horizon="short",
                score=float(adjusted_score),
                confidence=abs(float(adjusted_score) - float(threshold)),
            )
    except Exception:  # pragma: no cover - telemetry failures must not stop live scoring
        return


def command_signals(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    cfg = load_strategy()
    model_path = Path(getattr(args, "model", "data/model.json"))
    cache_path = Path(getattr(args, "cache", None) or _external_signal_cache_path(model_path))

    def run_once() -> tuple[int, ExternalSignalReport | None]:
        try:
            telemetry_path = None if getattr(args, "no_telemetry", False) or not cfg.telemetry_enabled else (
                getattr(args, "telemetry_db", None) or cfg.telemetry_db_path
            )
            report = collect_external_signals(
                symbol=runtime.symbol,
                cache_path=cache_path,
                ttl_seconds=max(0, int(getattr(args, "ttl", 300))),
                timeout_seconds=_clamp(float(getattr(args, "timeout", 3.0)), 0.1, 30.0),
                max_adjustment=_clamp(float(getattr(args, "max_adjustment", 0.04)), 0.0, 0.20),
                min_providers=max(0, min(120, int(getattr(args, "min_providers", 2)))),
                force_refresh=bool(getattr(args, "refresh", False)),
                compute_backend=str(getattr(args, "compute_backend", None) or runtime.compute_backend or "auto"),
                short_reaction_refresh_seconds=max(1, int(getattr(args, "short_reaction_refresh", 30))),
                news_provider_limit=max(
                    0,
                    int(getattr(args, "news_provider_limit", None) or cfg.external_signal_news_provider_limit),
                ),
                news_items_per_provider=max(
                    1,
                    int(getattr(args, "news_items_per_provider", None) or cfg.external_signal_news_items_per_provider),
                ),
                news_provider_parallelism=max(
                    1,
                    int(getattr(args, "provider_parallelism", None) or cfg.external_signal_provider_parallelism),
                ),
                news_provider_jitter_seconds=max(
                    0.0,
                    float(
                        getattr(args, "provider_jitter", None)
                        if getattr(args, "provider_jitter", None) is not None
                        else cfg.external_signal_provider_jitter_seconds
                    ),
                ),
                ollama_news_enabled=(
                    bool(getattr(args, "ollama_news", None))
                    if getattr(args, "ollama_news", None) is not None
                    else cfg.external_news_ai_enabled
                ),
                ollama_model=str(getattr(args, "ollama_model", None) or cfg.external_news_ai_model),
                ollama_url=str(getattr(args, "ollama_url", None) or cfg.external_news_ai_url),
                ollama_timeout_seconds=_clamp(
                    float(getattr(args, "ollama_timeout", None) or cfg.external_news_ai_timeout_seconds),
                    0.1,
                    30.0,
                ),
                telemetry_path=telemetry_path,
                source_grade_max_age_hours=_clamp(
                    float(
                        cfg.source_grade_max_age_hours
                        if getattr(args, "source_grade_max_age_hours", None) is None
                        else getattr(args, "source_grade_max_age_hours", None)
                    ),
                    0.0,
                    8760.0,
                ),
            )
        except Exception as exc:
            print(f"External signal collection failed: {exc}", file=sys.stderr)
            return 2, None
        if getattr(args, "json", False):
            print(json.dumps(report.asdict(), indent=2, sort_keys=True))
        else:
            print(render_external_signal_report(report))
        return (0 if report.status != "fail" else 2), report

    if not getattr(args, "loop", False):
        code, _report = run_once()
        return code

    iterations = max(0, int(getattr(args, "iterations", 0) or 0))
    sleep_seconds = float(getattr(args, "sleep", None) if getattr(args, "sleep", None) is not None else 60.0)
    jitter_seconds = float(
        getattr(args, "jitter", None)
        if getattr(args, "jitter", None) is not None
        else cfg.external_signal_poll_jitter_seconds
    )
    code = 0
    completed = 0
    try:
        while iterations <= 0 or completed < iterations:  # pragma: no branch
            code, _report = run_once()
            completed += 1
            if iterations > 0 and completed >= iterations:
                break
            time.sleep(_jittered_seconds(sleep_seconds, jitter_seconds))
    except KeyboardInterrupt:  # pragma: no cover - manual operator stop path
        print("Signal loop stopped.")
    return code


def command_source_grades(args: argparse.Namespace) -> int:
    cfg = load_strategy()
    try:
        run = grade_sources(
            db_path=getattr(args, "db", None) or cfg.telemetry_db_path,
            window_hours=float(getattr(args, "window_hours", None) or cfg.source_grading_window_hours),
            model=str(getattr(args, "ollama_model", None) or cfg.external_news_ai_model),
            ollama_enabled=(
                bool(getattr(args, "ollama", None))
                if getattr(args, "ollama", None) is not None
                else cfg.source_grading_enabled
            ),
            ollama_url=str(getattr(args, "ollama_url", None) or cfg.external_news_ai_url),
            ollama_timeout_seconds=_clamp(
                float(getattr(args, "ollama_timeout", None) or cfg.external_news_ai_timeout_seconds),
                0.1,
                120.0,
            ),
        )
    except Exception as exc:
        print(f"Source grading failed: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(run.asdict(), indent=2, sort_keys=True))
    else:
        print(render_source_grade_run(run))
    return 0 if run.status != "empty" else 2


def command_signals_benchmark(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    cfg = load_strategy()
    provider_limits = [max(0, int(value)) for value in (getattr(args, "provider_limit", None) or [30, 60])]
    parallelism_values = [max(1, int(value)) for value in (getattr(args, "parallelism", None) or [8, 16])]
    iterations = max(1, int(getattr(args, "iterations", 1) or 1))
    rows: list[dict[str, object]] = []
    worst_code = 0
    for provider_limit in provider_limits:
        for parallelism in parallelism_values:
            durations: list[float] = []
            fresh_counts: list[int] = []
            provider_counts: list[int] = []
            statuses: list[str] = []
            for _iteration in range(iterations):
                started = time.perf_counter()
                try:
                    report = collect_external_signals(
                        symbol=runtime.symbol,
                        cache_path=getattr(args, "cache", "data/signals/benchmark_external_signals.json"),
                        ttl_seconds=0,
                        timeout_seconds=_clamp(float(getattr(args, "timeout", 3.0)), 0.1, 30.0),
                        max_adjustment=cfg.external_signal_max_adjustment,
                        min_providers=cfg.external_signal_min_providers,
                        force_refresh=True,
                        compute_backend=runtime.compute_backend,
                        news_provider_limit=provider_limit,
                        news_provider_parallelism=parallelism,
                        news_provider_jitter_seconds=max(0.0, float(getattr(args, "provider_jitter", 0.0) or 0.0)),
                        ollama_news_enabled=(
                            bool(getattr(args, "ollama_news", None))
                            if getattr(args, "ollama_news", None) is not None
                            else cfg.external_news_ai_enabled
                        ),
                        ollama_model=str(getattr(args, "ollama_model", None) or cfg.external_news_ai_model),
                        ollama_url=str(getattr(args, "ollama_url", None) or cfg.external_news_ai_url),
                        ollama_timeout_seconds=_clamp(
                            float(getattr(args, "ollama_timeout", None) or cfg.external_news_ai_timeout_seconds),
                            0.1,
                            30.0,
                        ),
                        telemetry_path=None if getattr(args, "no_telemetry", False) or not cfg.telemetry_enabled else cfg.telemetry_db_path,
                        source_grade_max_age_hours=cfg.source_grade_max_age_hours,
                    )
                    status = report.status
                    fresh_counts.append(report.fresh_count)
                    provider_counts.append(report.provider_count)
                    if status == "fail":
                        worst_code = 2
                except Exception as exc:
                    status = "error"
                    worst_code = 2
                    fresh_counts.append(0)
                    provider_counts.append(provider_limit)
                    print(f"benchmark trial failed: {exc}", file=sys.stderr)
                durations.append((time.perf_counter() - started) * 1000.0)
                statuses.append(status)
            average_ms = sum(durations) / len(durations)
            row = {
                "provider_limit": provider_limit,
                "parallelism": parallelism,
                "iterations": iterations,
                "avg_ms": round(average_ms, 2),
                "max_ms": round(max(durations), 2),
                "avg_fresh": round(sum(fresh_counts) / len(fresh_counts), 2),
                "avg_providers": round(sum(provider_counts) / len(provider_counts), 2),
                "statuses": statuses,
            }
            rows.append(row)
    payload = {"command": "signals-benchmark", "trials": rows}
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Signal benchmark")
        for row in rows:
            print(
                f"- providers={row['provider_limit']} parallelism={row['parallelism']} "
                f"avg_ms={row['avg_ms']} max_ms={row['max_ms']} "
                f"fresh={row['avg_fresh']}/{row['avg_providers']} statuses={','.join(row['statuses'])}"
            )
    return worst_code


def command_live(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    cfg = load_strategy()
    try:
        compute_backend, _backend_info = _workflow_compute_backend(
            runtime,
            getattr(args, "compute_backend", None),
            workflow="live retraining",
        )
    except ValueError as exc:
        print(f"Live settings invalid: {exc}.", file=sys.stderr)
        return 2
    batch_size = max(1, int(getattr(args, "batch_size", 8192) or 8192))
    if getattr(args, "paper", False) and getattr(args, "live", False):
        print("Choose either --paper or --live, not both.")
        return 2
    leverage_override = getattr(args, "leverage", None)
    if leverage_override is not None and runtime.market_type != "futures":
        print("Leverage override is spot-inactive; spot runs at 1x.")
    external_override = getattr(args, "external_signals", None)
    if external_override is not None:
        cfg = StrategyConfig(**{**cfg.asdict(), "external_signals_enabled": bool(external_override)})
    model_path = Path(getattr(args, "model", "data/model.json"))

    if getattr(args, "live", False):
        effective_dry_run = False
    else:
        effective_dry_run = runtime.dry_run or getattr(args, "paper", False)
    if not effective_dry_run and int(getattr(args, "retrain_interval", 0) or 0) > 0:
        print("Authenticated live mode cannot retrain inside the live loop; use model-lab promotion first.", file=sys.stderr)
        return 2
    if not _allows_signed_execution(runtime):
        print("Real-money execution is disabled in this phase. Set testnet=true or demo=true to run.")
        return 2
    if not effective_dry_run and not _has_api_credentials(runtime):
        print(_credential_required_message("Authenticated live mode"), file=sys.stderr)
        return 2
    try:
        client = _build_client(runtime)
    except BinanceAPIError as exc:
        print(f"Live startup blocked: {exc}", file=sys.stderr)
        return 2
    if not effective_dry_run:
        try:
            budget_report = _ensure_api_budget_startup_safe(runtime, client)
        except BinanceAPIError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(summarize_api_budget(budget_report))
    sleep_seconds = max(0, int(getattr(args, "sleep", 0)))
    if not effective_dry_run and sleep_seconds < 1:
        print("Authenticated live mode uses minimum sleep=1s.")
        sleep_seconds = 1

    def live_sleep() -> None:
        delay = _jittered_seconds(sleep_seconds, cfg.external_signal_poll_jitter_seconds) if sleep_seconds > 0 else 0.0
        time.sleep(delay)

    quote_asset, base_asset = _fund_asset_labels(runtime)
    cash = max(0.0, _safe_float(getattr(runtime, "managed_usdc", 0.0)))
    exchange_account_snapshot: object | None = None
    if not effective_dry_run:
        try:
            exchange_account_snapshot = client.get_account()
        except BinanceAPIError as exc:
            print(f"Account balance check failed: {exc}", file=sys.stderr)
            return 2
        balances = _account_free_balances(exchange_account_snapshot, (quote_asset, base_asset))
        available_quote = max(0.0, _safe_float(balances.get(quote_asset, 0.0)))
        if cash <= 0.0:
            print(f"Authenticated live mode requires a positive {quote_asset} trading cap from Funds.", file=sys.stderr)
            return 2
        if available_quote <= 0.0:
            print(f"Authenticated live mode requires available {quote_asset} on the exchange account.", file=sys.stderr)
            return 2
        if cash > available_quote:
            cash = available_quote
            print(f"Authenticated live cash capped to exchange free {quote_asset}={cash:.2f}.")

    model, model_error, model_notice = _load_live_start_model(
        model_path,
        cfg,
        effective_dry_run=effective_dry_run,
        require_model_candidate_search=not effective_dry_run,
        require_accelerator_evidence=not effective_dry_run and _backend_info.kind != "cpu",
        require_live_data_evidence=not effective_dry_run,
        require_microstructure_evidence=not effective_dry_run and runtime.market_type == "futures",
        expected_symbol=runtime.symbol,
        expected_market_type=runtime.market_type,
        expected_interval=runtime.interval,
    )
    if model_error is not None:
        print(model_error, file=sys.stderr)
        return 2
    if model is not None:
        protected = ("leverage",) if leverage_override is not None else ()
        cfg = apply_model_strategy_overrides(cfg, model, protected_keys=protected)
    if leverage_override is not None and runtime.market_type == "futures":
        cfg = StrategyConfig(**{**cfg.asdict(), "leverage": max(1.0, float(leverage_override))})
    if external_override is not None:
        cfg = StrategyConfig(**{**cfg.asdict(), "external_signals_enabled": bool(external_override)})
    if model_notice is not None:
        print(model_notice, file=sys.stderr)

    constraints = _resolve_symbol_constraints(runtime, client)
    if constraints is None and not effective_dry_run:
        print(
            f"Authenticated live mode requires {runtime.symbol} exchange filters "
            "before any order is allowed.",
            file=sys.stderr,
        )
        return 2

    try:
        if effective_dry_run:
            cfg, commission_assumption = apply_offline_commission_floor(
                cfg,
                market_type=runtime.market_type,
                symbol=runtime.symbol,
            )
        else:
            cfg, commission_assumption, _commission_rates = apply_verified_commission_rate(
                cfg,
                client=client,
                symbol=runtime.symbol,
            )
    except (BinanceAPIError, ValueError) as exc:
        print(f"Live startup blocked: commission-rate verification failed: {exc}", file=sys.stderr)
        return 2
    model_fee_error = _model_taker_fee_compatibility_error(
        model,
        commission_assumption.modeled_taker_fee_bps,
    )
    if not effective_dry_run and model_fee_error is not None:
        print(f"Live startup blocked: {model_fee_error}", file=sys.stderr)
        return 2

    leverage = _resolve_futures_leverage(runtime, cfg)
    position_notional = 0.0
    position_side = 0
    entry_price = 0.0
    margin_used = 0.0
    qty = 0.0
    wait_ticks = cfg.cooldown_minutes
    min_position_hold_bars = max(0, int(getattr(cfg, "min_position_hold_bars", 0) or 0))
    flat_signal_exit_grace_bars = max(0, int(getattr(cfg, "flat_signal_exit_grace_bars", 0) or 0))
    max_position_hold_bars = max(0, int(getattr(cfg, "max_position_hold_bars", 0) or 0))
    entry_market_timestamp_ms = 0
    flat_signal_streak = 0
    cooldown_left = 0
    unpredictability_cooldown_left = 0
    if leverage > MAX_AUTONOMOUS_LEVERAGE:
        leverage = MAX_AUTONOMOUS_LEVERAGE
    elif leverage < 1.0:
        leverage = 1.0
    position_leverage = leverage
    equity_peak = cash
    max_drawdown_seen = 0.0
    risk_policy = build_risk_policy_report(
        runtime,
        cfg,
        effective_dry_run=effective_dry_run,
        leverage=leverage,
    )
    if not risk_policy.allowed:
        print(render_risk_policy_report(risk_policy), file=sys.stderr)
        return 2

    mode_label = "paper" if effective_dry_run else "live"
    print(f"Starting {mode_label} loop for {args.steps} steps on {runtime.symbol} {runtime.interval} [{runtime.market_type}]")
    if runtime.market_type == "futures":
        print(f"effective leverage: {leverage:.1f}x")
    print(
        "taker fee: "
        f"{commission_assumption.modeled_taker_fee_bps:.4f} bps "
        f"source={commission_assumption.source} verified={str(commission_assumption.verified).lower()}"
    )
    if cfg.external_signals_enabled:
        print(
            "external signals: enabled "
            f"max_adjust={cfg.external_signal_max_adjustment:.3f} "
            f"min_providers={cfg.external_signal_min_providers}"
        )

    fee_rate = max(0.0, cfg.taker_fee_bps) / 10_000.0
    slippage = max(0.0, cfg.slippage_bps) / 10_000.0
    max_daily_trades = int(cfg.max_trades_per_day)
    max_daily_trades = max(max_daily_trades, 0)
    daily_trade_count: dict[int, int] = {}
    max_open_positions = int(cfg.max_open_positions)
    live_run = build_live_run_payload(
        runtime_public=_public_runtime_payload(runtime),
        strategy=cfg,
        steps_total=int(args.steps),
        market=runtime.market_type,
        symbol=runtime.symbol,
        model_path=model_path,
        model=model,
        starting_cash=cash,
        external_signal_cache=_external_signal_cache_path(model_path),
        risk_policy=risk_policy,
    )
    live_events = cast(list[dict[str, object]], live_run["events"])
    if risk_policy.warning_count:
        print(f"risk policy warnings: {risk_policy.warning_count}")
    live_position_store = PositionsStore() if not effective_dry_run else None
    current_position_id = ""
    current_open_client_order_id = ""
    current_open_exchange_order_id = ""
    current_opened_at_ms = 0
    entry_fee_paid = 0.0

    def persist_live_startup_block(status: str, reason: str, details: Mapping[str, object] | None = None) -> None:
        event: dict[str, object] = {"step": 0, "status": status, "reason": reason}
        if details is not None:
            event["details"] = dict(details)
        live_events.append(event)
        live_run["result"] = {
            "status": status,
            "reason": reason,
            "steps_executed": 0,
            "entries": 0,
            "closes": 0,
            "skipped_entries": 0,
            "model_loads": 0 if model is None else 1,
            "drawdown_seen": 0.0,
            "ending_cash": float(cash),
            "ending_cash_is_estimate": True,
            "equity_peak": float(cash),
            "drawdown_limit": float(cfg.max_drawdown_limit),
        }
        _persist_run_artifact("live", model_path.parent, live_run)

    if not effective_dry_run:
        if not isinstance(exchange_account_snapshot, Mapping):
            reason = "signed account payload was not a mapping"
            print(f"Authenticated live startup blocked: {reason}.", file=sys.stderr)
            persist_live_startup_block("startup_reconciliation_error", reason)
            return 2
        position_store = live_position_store if live_position_store is not None else PositionsStore()
        try:
            reconciliation_report = reconcile_account_positions(
                exchange_account_snapshot,
                runtime,
                position_store,
                quantity_tolerance=1e-8,
            )
        except (OSError, ValueError) as exc:
            reason = f"reconciliation failed: {exc}"
            print(f"Authenticated live startup blocked: {reason}", file=sys.stderr)
            persist_live_startup_block("startup_reconciliation_error", reason)
            return 2
        if not reconciliation_report.ok:
            reason = "exchange exposure does not match the bot-owned local ledger"
            print(
                "Authenticated live startup blocked: "
                f"{reason}. Run `simple-ai-trading reconcile` and resolve the mismatch before live trading.",
                file=sys.stderr,
            )
            persist_live_startup_block(
                "startup_reconciliation_mismatch",
                reason,
                reconciliation_report.asdict(),
            )
            return 2

        ledger_positions = [
            position
            for position in position_store.load_open()
            if not position.dry_run and str(position.symbol).upper() == runtime.symbol.upper()
        ]
        if len(ledger_positions) > 1:
            reason = f"multiple bot-owned ledger positions for {runtime.symbol}"
            print(f"Authenticated live startup blocked: {reason}.", file=sys.stderr)
            persist_live_startup_block("startup_reconciliation_mismatch", reason, reconciliation_report.asdict())
            return 2
        if ledger_positions:
            ledger_position = ledger_positions[0]
            ownership_rejection = bot_ownership_rejection_reason(ledger_position)
            if ownership_rejection is not None:
                reason = (
                    f"ledger position {ledger_position.id} lacks bot order ownership proof "
                    f"({ownership_rejection})"
                )
                print(f"Authenticated live startup blocked: {reason}.", file=sys.stderr)
                persist_live_startup_block("startup_reconciliation_mismatch", reason, reconciliation_report.asdict())
                return 2
            position_side = 1 if str(ledger_position.side).upper() == "LONG" else -1
            qty = max(0.0, float(ledger_position.qty))
            entry_price = max(0.0, float(ledger_position.entry_price))
            position_notional = position_side * max(0.0, float(ledger_position.notional or qty * entry_price))
            ledger_leverage = _safe_float(getattr(ledger_position, "leverage", leverage))
            if runtime.market_type == "spot":
                position_leverage = 1.0
            elif math.isfinite(ledger_leverage) and ledger_leverage > 0.0:
                position_leverage = max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, ledger_leverage))
            else:
                position_leverage = leverage
            margin_used = (
                min(cash, abs(position_notional))
                if runtime.market_type == "spot"
                else min(cash, abs(position_notional) / max(1.0, position_leverage))
            )
            cash = max(0.0, cash - margin_used)
            print(
                "Resuming bot-owned ledger position: "
                f"{'long' if position_side > 0 else 'short'} qty={qty:.8f} entry={entry_price:.2f}"
            )
            live_events.append(
                {
                    "step": 0,
                    "status": "resume_bot_ledger_position",
                    "position_id": ledger_position.id,
                    "market": ledger_position.market_type,
                    "direction": int(position_side),
                    "qty": float(qty),
                    "entry_price": float(entry_price),
                    "notional": float(position_notional),
                    "margin": float(margin_used),
                    "leverage": float(position_leverage),
                    "cash_after_resume": float(cash),
                    "open_client_order_id": ledger_position.open_client_order_id,
                }
            )
            current_position_id = ledger_position.id
            current_open_client_order_id = ledger_position.open_client_order_id
            current_open_exchange_order_id = ledger_position.open_exchange_order_id
            current_opened_at_ms = int(ledger_position.opened_at_ms)
            entry_market_timestamp_ms = int(ledger_position.opened_at_ms)
    drawdown_limit = float(cfg.max_drawdown_limit)
    halt_reason = "completed"
    steps_executed = 0
    entries = 0
    closes = 0
    skipped_entries = 0
    model_loads = 0 if model is None else 1
    exit_code = 0
    network_errors = 0
    max_network_errors = max(1, int(getattr(cfg, "max_network_errors", 1) or 1))
    recovery_pending = False

    def _next_source_grade_time_ms(after_ms: int) -> int:  # pragma: no cover - hourly live-loop scheduler
        return after_ms + int(_jittered_seconds(cfg.source_grading_interval_seconds, cfg.external_signal_poll_jitter_seconds) * 1000)

    next_source_grade_ms = _next_source_grade_time_ms(int(time.time() * 1000))

    def maybe_grade_sources(step: int) -> None:  # pragma: no cover - exercised by long-running live sessions
        nonlocal next_source_grade_ms
        if not cfg.telemetry_enabled or not cfg.source_grading_enabled:
            return
        current_ms = int(time.time() * 1000)
        if current_ms < next_source_grade_ms:
            return
        try:
            grade_run = grade_sources(
                db_path=cfg.telemetry_db_path,
                window_hours=cfg.source_grading_window_hours,
                model=cfg.external_news_ai_model,
                ollama_enabled=bool(cfg.source_grading_enabled),
                ollama_url=cfg.external_news_ai_url,
                ollama_timeout_seconds=cfg.external_news_ai_timeout_seconds,
            )
            live_events.append(
                {
                    "step": int(step),
                    "status": "source_grading",
                    "graded_sources": int(grade_run.graded_sources),
                    "ai_status": grade_run.ai_status,
                    "warnings": list(grade_run.warnings),
                }
            )
            print(
                f"step {step:>2}: source grades "
                f"graded={grade_run.graded_sources} ai={grade_run.ai_status}"
            )
        except Exception as exc:
            live_events.append({"step": int(step), "status": "source_grading_error", "error": str(exc)})
            print(f"step {step:>2}: source grading unavailable: {exc}", file=sys.stderr)
        finally:
            next_source_grade_ms = _next_source_grade_time_ms(current_ms)

    def record_order_error(step: int, side: str, size: float, exc: BinanceAPIError) -> None:
        nonlocal halt_reason, exit_code
        print(f"order error: {exc}", file=sys.stderr)
        halt_reason = "order_error"
        exit_code = 2
        live_events.append(
            {
                "step": step,
                "status": "order_error",
                "side": side,
                "size": float(size),
                "error": str(exc),
            }
        )

    def record_signed_live_open(
        *,
        position_id: str,
        side_sign: int,
        filled_qty: float,
        fill_price: float,
        notional: float,
        leverage_used: float,
        order_response: object,
        open_client_order_id: str,
    ) -> None:
        nonlocal current_position_id, current_open_client_order_id, current_open_exchange_order_id, current_opened_at_ms
        if live_position_store is None or not position_id:
            return
        current_position_id = position_id
        current_open_client_order_id = _order_response_text(order_response, "clientOrderId", "origClientOrderId") or open_client_order_id
        current_open_exchange_order_id = _order_response_text(order_response, "orderId")
        current_opened_at_ms = int(time.time() * 1000)
        live_position_store.record_open(
            OpenPosition(
                id=position_id,
                symbol=runtime.symbol,
                market_type=runtime.market_type,
                side="LONG" if side_sign > 0 else "SHORT",
                qty=max(0.0, float(filled_qty)),
                entry_price=max(0.0, float(fill_price)),
                leverage=float(leverage_used),
                opened_at_ms=current_opened_at_ms,
                notional=abs(float(notional)),
                strategy_profile="live-cli",
                objective="live",
                dry_run=False,
                stop_loss_pct=float(cfg.stop_loss_pct),
                take_profit_pct=float(cfg.take_profit_pct),
                open_client_order_id=current_open_client_order_id,
                open_exchange_order_id=current_open_exchange_order_id,
                exchange_status=_order_response_text(order_response, "status") or ("FILLED" if filled_qty > 0.0 else "accepted"),
            )
        )

    def record_signed_live_close(
        *,
        reason: str,
        closed_qty: float,
        fill_price: float,
        realized_pnl: float,
        close_ratio: float,
        close_fee: float,
        order_response: object,
        close_client_order_id: str | None,
    ) -> None:
        nonlocal current_position_id, current_open_client_order_id, current_open_exchange_order_id, current_opened_at_ms, entry_fee_paid
        if live_position_store is None or not current_position_id:
            return
        open_position = live_position_store.find_open(current_position_id)
        if open_position is None:
            live_events.append(
                {
                    "step": 0,
                    "status": "ledger_close_missing_open_position",
                    "position_id": current_position_id,
                    "reason": reason,
                }
            )
            return
        fee_total = max(0.0, entry_fee_paid * max(0.0, min(1.0, close_ratio))) + max(0.0, float(close_fee))
        close_client = _order_response_text(order_response, "clientOrderId", "origClientOrderId") or str(close_client_order_id or "")
        close_exchange = _order_response_text(order_response, "orderId")
        if close_ratio >= 0.999:
            live_position_store.record_close(
                ClosedTrade(
                    id=open_position.id,
                    symbol=open_position.symbol,
                    market_type=open_position.market_type,
                    side=open_position.side,
                    qty=max(0.0, float(closed_qty)),
                    entry_price=float(open_position.entry_price),
                    exit_price=max(0.0, float(fill_price)),
                    leverage=float(open_position.leverage),
                    opened_at_ms=int(open_position.opened_at_ms),
                    closed_at_ms=int(time.time() * 1000),
                    realized_pnl=float(realized_pnl) - fee_total,
                    realized_pnl_pct=(
                        (float(realized_pnl) - fee_total) / max(1e-12, abs(float(open_position.entry_price) * float(closed_qty)))
                    ),
                    fees=fee_total,
                    reason=reason,
                    strategy_profile=open_position.strategy_profile,
                    objective=open_position.objective,
                    dry_run=False,
                    owner=open_position.owner,
                    open_client_order_id=open_position.open_client_order_id,
                    open_exchange_order_id=open_position.open_exchange_order_id,
                    close_client_order_id=close_client,
                    close_exchange_order_id=close_exchange,
                    exchange_status=_order_response_text(order_response, "status") or ("FILLED" if closed_qty > 0.0 else "accepted"),
                )
            )
            current_position_id = ""
            current_open_client_order_id = ""
            current_open_exchange_order_id = ""
            current_opened_at_ms = 0
            entry_fee_paid = 0.0
            return
        remaining_qty = max(0.0, float(open_position.qty) - max(0.0, float(closed_qty)))
        entry_fee_paid = max(0.0, entry_fee_paid * (1.0 - max(0.0, min(1.0, close_ratio))))
        live_position_store.record_open(
            replace(
                open_position,
                qty=remaining_qty,
                notional=remaining_qty * max(0.0, float(open_position.entry_price)),
                exchange_status="partial",
            )
        )

    def finish_recovery_observation(step: int, price: float, reason: str) -> None:
        nonlocal network_errors, recovery_pending, halt_reason, exit_code
        observed_errors = int(network_errors)
        cooldown = max(0, int(getattr(cfg, "recovery_cooldown_seconds", 0) or 0))
        live_events.append(
            {
                "step": step,
                "status": "recovery_observation",
                "reason": reason,
                "price": float(price),
                "network_errors": observed_errors,
                "max_network_errors": int(max_network_errors),
                "cooldown_seconds": int(cooldown),
            }
        )
        print(
            f"step {step:>2}: recovery observation after {observed_errors} market errors; "
            "new entries paused"
        )
        recovery_pending = False
        network_errors = 0
        if halt_reason == "market_recovery_pending":
            halt_reason = "completed"
            exit_code = 0

    def recovery_sleep() -> None:
        cooldown = max(0, int(getattr(cfg, "recovery_cooldown_seconds", 0) or 0))
        if cooldown > sleep_seconds:
            time.sleep(float(cooldown))
        else:
            live_sleep()

    for i in range(args.steps):
        if unpredictability_cooldown_left > 0:
            unpredictability_cooldown_left -= 1
        try:
            candles = client.get_klines(runtime.symbol, runtime.interval, limit=max(cfg.model_lookback, 300))
        except BinanceAPIError as exc:
            network_errors += 1
            recovery_pending = True
            halt_reason = "market_recovery_pending"
            exit_code = 2
            status = "market_error_retry_limit" if network_errors >= max_network_errors else "market_error_retry"
            print(
                f"market error: {exc}; retrying "
                f"({network_errors}/{max_network_errors}) before any new entry",
                file=sys.stderr,
            )
            live_events.append(
                {
                    "step": i + 1,
                    "status": status,
                    "error": str(exc),
                    "network_errors": int(network_errors),
                    "max_network_errors": int(max_network_errors),
                    "recovery_pending": True,
                }
            )
            live_sleep()
            continue

        steps_executed += 1
        recovery_observation_active = bool(recovery_pending)

        rows = _live_rows_for_model(candles, cfg, model, compute_backend=compute_backend)
        training_rows = (
            _readiness_model_rows(candles, cfg, model, compute_backend=compute_backend)
            if model is not None
            else _build_model_rows(candles, cfg, compute_backend=compute_backend)
        )
        if not rows:
            print("not enough historical data for live signal")
            live_events.append({"step": i + 1, "status": "no_rows"})
            live_sleep()
            continue

        live_events.append({"step": i + 1, "status": "rows", "count": len(rows)})

        retrain_interval = getattr(args, "retrain_interval", 0)
        retrain_window = getattr(args, "retrain_window", 300)
        retrain_min_rows = getattr(args, "retrain_min_rows", 240)
        if retrain_min_rows <= 0:
            retrain_min_rows = max(1, 240)
        if retrain_window <= 0:
            retrain_window = max(1, 300)
        retrain_interval = max(retrain_interval, 0)

        previous_model = model
        model = _build_live_model(
            training_rows,
            model=model,
            retrain_every=retrain_interval,
            step=i + 1,
            cfg=cfg,
            retrain_window=retrain_window,
            retrain_min_rows=retrain_min_rows,
            model_feature_signature=_live_model_feature_signature(model, cfg),
            compute_backend=compute_backend,
            batch_size=batch_size,
        )
        if previous_model is None and model is not None:
            model_loads += 1
            if model is not None and model.__class__.__name__ != "TrainedModel":
                model_signature = None
            else:
                model_signature = getattr(model, "feature_signature", None)
                live_run["model_signature"] = str(model_signature) if model_signature else None
        elif previous_model is not None and model is not None and previous_model is not model:
            model_loads += 1
            model_signature = getattr(model, "feature_signature", None)
            live_run["model_signature"] = str(model_signature) if model_signature else None

        if model is None:
            if not training_rows:
                print("not enough labeled historical data to train a live model")
                live_events.append({"step": i + 1, "status": "no_training_rows"})
                live_sleep()
                continue
            model = train(
                training_rows,
                epochs=40,
                feature_signature=_live_model_feature_signature(model, cfg),
                compute_backend=compute_backend,
                batch_size=batch_size,
            )
            model_loads += 1
        latest = rows[-1]
        regime_window_size = max(8, min(len(rows), int(cfg.liquidity_lookback_bars)))
        regime_evidence = classify_market_regime(rows[-regime_window_size:])
        regime_score = market_regime_unpredictability(
            regime_evidence.dominant_regime,
            regime_evidence.confidence,
            regime_evidence.notes,
        )
        regime_limit = float(cfg.max_regime_unpredictability)
        if (
            regime_score > regime_limit
            and regime_unpredictability_requires_cooldown(regime_score, regime_limit)
        ):
            unpredictability_cooldown_left = max(
                unpredictability_cooldown_left,
                max(1, int(cfg.unpredictability_cooldown_minutes)),
            )
        regime_cooldown_active = unpredictability_cooldown_left > 0
        if regime_cooldown_active or regime_score > regime_limit:
            live_events.append(
                {
                    "step": i + 1,
                    "status": "regime_unpredictability_gate",
                    "regime": regime_evidence.dominant_regime,
                    "confidence": float(regime_evidence.confidence),
                    "score": float(regime_score),
                    "limit": float(regime_limit),
                    "cooldown_left": int(unpredictability_cooldown_left),
                    "notes": list(regime_evidence.notes),
                }
            )
            relation = ">" if regime_score > regime_limit else "<="
            print(
                f"step {i + 1:>2}: regime gate {regime_evidence.dominant_regime} "
                f"score={regime_score:.2f}{relation}{regime_limit:.2f} cooldown={unpredictability_cooldown_left}"
            )
        if hasattr(model, "feature_means") and hasattr(model, "feature_stds"):
            try:
                drift = feature_drift_report([latest], model)
            except ValueError as exc:
                if not effective_dry_run:
                    print(f"Live feature drift check failed: {exc}", file=sys.stderr)
                    halt_reason = "feature_drift_check_failed"
                    exit_code = 2
                    live_events.append({"step": i + 1, "status": halt_reason, "error": str(exc)})
                    break
                drift = None
            if drift is not None:
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "feature_drift",
                        "drift_status": drift.status,
                        "max_abs_z": float(drift.max_abs_z),
                        "outlier_fraction": float(drift.outlier_fraction),
                        "warnings": list(drift.warnings),
                    }
                )
                if drift.status == "fail":
                    print(
                        f"step {i + 1:>2}: feature drift block "
                        f"max_z={drift.max_abs_z:.2f} outliers={drift.outlier_fraction:.1%}"
                    )
                    live_sleep()
                    continue
                if drift.status == "warn":
                    print(
                        f"step {i + 1:>2}: feature drift warning "
                        f"max_z={drift.max_abs_z:.2f} outliers={drift.outlier_fraction:.1%}"
                    )
        try:
            raw_score = model.predict_proba(latest.features)
            threshold = model_decision_threshold(model, cfg.signal_threshold)
            score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        except ValueError as exc:
            if not effective_dry_run:
                print(f"Live model incompatible with current rows: {exc}", file=sys.stderr)
                halt_reason = "model_incompatible"
                exit_code = 2
                live_events.append({"step": i + 1, "status": "model_incompatible", "error": str(exc)})
                break
            print(f"paper model incompatible; retraining: {exc}", file=sys.stderr)
            if not training_rows:
                print("not enough labeled historical data to retrain a live model", file=sys.stderr)
                halt_reason = "model_incompatible"
                exit_code = 2
                live_events.append({"step": i + 1, "status": "model_incompatible", "error": "no labeled training rows"})
                break
            model = train(
                training_rows,
                epochs=40,
                feature_signature=_live_model_feature_signature(model, cfg),
                compute_backend=compute_backend,
                batch_size=batch_size,
            )
            model_loads += 1
            raw_score = model.predict_proba(latest.features)
            threshold = model_decision_threshold(model, cfg.signal_threshold)
            score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        decision_cfg = cfg
        if cfg.external_signals_enabled:
            score_before_external = score
            try:
                external_report = collect_external_signals(
                    symbol=runtime.symbol,
                    cache_path=_external_signal_cache_path(model_path),
                    ttl_seconds=cfg.external_signal_ttl_seconds,
                    timeout_seconds=cfg.external_signal_timeout_seconds,
                    max_adjustment=cfg.external_signal_max_adjustment,
                    min_providers=cfg.external_signal_min_providers,
                    compute_backend=runtime.compute_backend,
                    short_reaction_refresh_seconds=cfg.external_signal_short_reaction_refresh_seconds,
                    news_provider_limit=cfg.external_signal_news_provider_limit,
                    news_items_per_provider=cfg.external_signal_news_items_per_provider,
                    news_provider_parallelism=cfg.external_signal_provider_parallelism,
                    news_provider_jitter_seconds=cfg.external_signal_provider_jitter_seconds,
                    ollama_news_enabled=cfg.external_news_ai_enabled,
                    ollama_model=cfg.external_news_ai_model,
                    ollama_url=cfg.external_news_ai_url,
                    ollama_timeout_seconds=cfg.external_news_ai_timeout_seconds,
                    telemetry_path=cfg.telemetry_db_path if cfg.telemetry_enabled else None,
                    source_grade_max_age_hours=cfg.source_grade_max_age_hours,
                )
                score, decision_cfg, applied_adjustment = _apply_external_signal_to_score(
                    score,
                    cfg,
                    external_report,
                )
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "external_signals",
                        "score_before": float(score_before_external),
                        "score_after": float(score),
                        "applied_adjustment": float(applied_adjustment),
                        "risk_multiplier": float(external_report.risk_multiplier),
                        "fresh_count": int(external_report.fresh_count),
                        "provider_count": int(external_report.provider_count),
                        "short_term_score": float(external_report.short_term_score),
                        "medium_term_score": float(external_report.medium_term_score),
                        "long_term_score": float(external_report.long_term_score),
                        "reaction_required": bool(external_report.reaction_required),
                        "reaction_reason": external_report.reaction_reason,
                        "news_backend_kind": external_report.news_backend_kind,
                        "news_ai_status": external_report.news_ai_status,
                        "news_ai_model": external_report.news_ai_model,
                        "news_ai_latency_ms": int(external_report.news_ai_latency_ms),
                        "report": external_report.asdict(),
                    }
                )
                print(
                    f"step {i + 1:>2}: external signals "
                    f"providers={external_report.fresh_count}/{external_report.provider_count} "
                    f"adj={applied_adjustment:+.4f} risk={external_report.risk_multiplier:.3f} "
                    f"short={external_report.short_term_score:+.3f} "
                    f"reaction={'yes' if external_report.reaction_required else 'no'}"
                )
            except Exception as exc:
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "external_signals_error",
                        "error": str(exc),
                    }
                )
                print(f"step {i + 1:>2}: external signals unavailable: {exc}", file=sys.stderr)
        base_long_threshold, base_short_threshold = model_direction_thresholds(
            model,
            decision_cfg.signal_threshold,
            market_type=runtime.market_type,
        )
        side_thresholds_explicit = (
            getattr(model, "long_decision_threshold", None) is not None
            or getattr(model, "short_decision_threshold", None) is not None
        )
        base_threshold = _directional_threshold_confidence(base_long_threshold, base_short_threshold, threshold)
        liquidity_adjustment = liquidity_session_adjustment(rows, len(rows) - 1, decision_cfg, base_threshold)
        threshold_add = float(liquidity_adjustment.threshold) - float(base_threshold)
        effective_long_threshold, effective_short_threshold = _adjust_directional_thresholds(
            base_long_threshold,
            base_short_threshold,
            threshold_add,
        )
        effective_threshold = _directional_threshold_confidence(
            effective_long_threshold,
            effective_short_threshold,
            threshold,
        )
        _record_model_telemetry(
            cfg,
            step=i + 1,
            row=latest,
            raw_score=raw_score,
            adjusted_score=score,
            threshold=effective_threshold,
            model=model,
            runtime=runtime,
        )
        direction = _score_to_direction(
            score,
            decision_cfg,
            runtime.market_type,
            effective_long_threshold,
            short_threshold=effective_short_threshold,
            side_thresholds_explicit=side_thresholds_explicit,
        )
        meta_threshold = (
            1.0 - effective_short_threshold
            if runtime.market_type == "futures" and direction < 0 and effective_short_threshold is not None
            else effective_threshold
        )
        base_meta_decision = apply_meta_label_policy(
            getattr(model, "meta_label_policy", {}),
            adjusted_probability=score,
            threshold=meta_threshold,
            side=direction,
            market_type=runtime.market_type,
        )
        meta_decision = apply_liquidity_session_meta(base_meta_decision, liquidity_adjustment) if direction != 0 else base_meta_decision
        if liquidity_adjustment.active:
            live_events.append(
                {
                    "step": i + 1,
                    "status": "liquidity_session_guard",
                    "threshold": float(effective_threshold),
                    "size_multiplier": float(liquidity_adjustment.size_multiplier),
                    "low_liquidity": bool(liquidity_adjustment.low_liquidity),
                    "low_dynamic_session": bool(liquidity_adjustment.low_dynamic_session),
                    "reasons": list(liquidity_adjustment.reasons),
                }
            )

        # cooldown reduces immediate flip-flopping in choppy conditions
        if cooldown_left > 0:
            if runtime.market_type == "spot":
                direction = 0
            cooldown_left -= 1

        price = latest.close
        day = _safe_day_ms(latest.timestamp)
        if position_side == 0 and direction != 0:
            if recovery_observation_active:
                current_drawdown = (equity_peak - cash) / equity_peak if equity_peak else 0.0
                entry_risk = assess_entry_risk(
                    direction=direction,
                    position_side=position_side,
                    max_open_positions=max_open_positions,
                    max_daily_trades=max_daily_trades,
                    daily_trade_count=daily_trade_count.get(day, 0),
                    cash=cash,
                    price=price,
                    drawdown=current_drawdown,
                    drawdown_limit=drawdown_limit,
                    network_errors=network_errors,
                    max_network_errors=max_network_errors,
                    recovery_pending=True,
                    regime=regime_evidence.dominant_regime,
                    regime_confidence=regime_evidence.confidence,
                    regime_notes=regime_evidence.notes,
                    regime_unpredictability_score=regime_score,
                    max_regime_unpredictability=regime_limit,
                    regime_cooldown_active=regime_cooldown_active,
                )
                message, event = _live_entry_risk_skip(
                    step=i + 1,
                    day=day,
                    score=score,
                    entry_risk=entry_risk,
                    max_daily_trades=max_daily_trades,
                    max_open_positions=max_open_positions,
                )
                print(message)
                skipped_entries += 1
                live_events.append(event)
                finish_recovery_observation(i + 1, price, "entry-risk")
                maybe_grade_sources(i + 1)
                recovery_sleep()
                continue
            if meta_decision.enabled and meta_decision.size_multiplier <= 0.0:
                print(f"step {i + 1:>2}: meta-label skipped entry ({meta_decision.reason})")
                skipped_entries += 1
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "skip_meta_label",
                        "score": float(score),
                        "direction": int(direction),
                        "meta_label": meta_decision.asdict(),
                    }
                )
                live_sleep()
                continue
            entry_cfg = (
                _strategy_with_size_multiplier(decision_cfg, meta_decision.size_multiplier)
                if meta_decision.enabled and meta_decision.action == "downsize"
                else decision_cfg
            )
            current_drawdown = (equity_peak - cash) / equity_peak if equity_peak else 0.0
            entry_risk = assess_entry_risk(
                direction=direction,
                position_side=position_side,
                max_open_positions=max_open_positions,
                max_daily_trades=max_daily_trades,
                daily_trade_count=daily_trade_count.get(day, 0),
                cash=cash,
                price=price,
                drawdown=current_drawdown,
                drawdown_limit=drawdown_limit,
                regime=regime_evidence.dominant_regime,
                regime_confidence=regime_evidence.confidence,
                regime_notes=regime_evidence.notes,
                regime_unpredictability_score=regime_score,
                max_regime_unpredictability=regime_limit,
                regime_cooldown_active=regime_cooldown_active,
            )
            if not entry_risk.allowed:
                message, event = _live_entry_risk_skip(
                    step=i + 1,
                    day=day,
                    score=score,
                    entry_risk=entry_risk,
                    max_daily_trades=max_daily_trades,
                    max_open_positions=max_open_positions,
                )
                print(message)
                skipped_entries += 1
                live_events.append(event)
                live_sleep()
                continue

            notional, qty = _build_order_notional(
                cash,
                price,
                entry_cfg,
                runtime.market_type,
                leverage,
                client,
                constraints=constraints,
            )
            if notional <= 0:
                print(f"step {i + 1:>2}: skipped entry due to order constraints")
                skipped_entries += 1
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "skip_constraints",
                        "score": float(score),
                        "notional": float(notional),
                    }
                )
                live_sleep()
                continue

            side_sign = 1 if direction > 0 else -1
            side = "BUY" if side_sign > 0 else "SELL"
            fill = price * (1.0 + side_sign * slippage)
            if fill <= 0:
                live_sleep()
                continue
            notional = qty * fill
            try:
                entry_leverage = _entry_leverage_for_notional(
                    client,
                    runtime,
                    leverage,
                    abs(notional),
                    effective_dry_run=effective_dry_run,
                )
            except BinanceAPIError as exc:
                record_order_error(i + 1, side, qty, exc)
                break

            if runtime.market_type == "spot":
                margin = min(cash, abs(notional))
            else:
                margin = min(cash, abs(notional) / entry_leverage)

            fee = abs(notional) * fee_rate
            total = margin + fee
            if total > cash:
                print(f"step {i + 1:>2}: insufficient cash for leverage-adjusted entry")
                skipped_entries += 1
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "skip_insufficient_cash_pre_fill",
                        "score": float(score),
                    }
                )
                live_sleep()
                continue

            live_position_id = new_position_id() if not effective_dry_run else ""
            open_client_order_id = bot_client_order_id(live_position_id, "open") if live_position_id else None
            try:
                order_response = _paper_or_live_order(
                    client,
                    runtime,
                    entry_cfg,
                    side=side,
                    size=qty,
                    dry_run=effective_dry_run,
                    leverage=entry_leverage,
                    notional=abs(notional),
                    client_order_id=open_client_order_id,
                )
            except BinanceAPIError as exc:
                record_order_error(i + 1, side, qty, exc)
                break

            filled_qty, fill, filled_notional, fill_source = _resolved_order_fill_details(
                client,
                runtime,
                order_response,
                fallback_qty=qty,
                fallback_price=fill,
                dry_run=effective_dry_run,
            )
            if filled_qty <= 0.0 or fill <= 0.0:
                record_order_error(i + 1, side, qty, BinanceAPIError("Order response did not include executed quantity"))
                break
            qty = abs(filled_qty)
            notional = filled_notional if filled_notional > 0.0 else qty * fill
            fee = abs(notional) * fee_rate
            margin = min(cash, abs(notional)) if runtime.market_type == "spot" else min(cash, abs(notional) / entry_leverage)
            total = margin + fee

            cash = max(0.0, cash - total)
            position_side = direction
            position_notional = direction * notional
            qty = abs(qty)
            entry_price = fill
            entry_market_timestamp_ms = int(latest.timestamp)
            flat_signal_streak = 0
            position_leverage = entry_leverage
            margin_used = margin
            entry_fee_paid = fee
            daily_trade_count[day] = daily_trade_count.get(day, 0) + 1
            record_signed_live_open(
                position_id=live_position_id,
                side_sign=side_sign,
                filled_qty=qty,
                fill_price=fill,
                notional=notional,
                leverage_used=position_leverage,
                order_response=order_response,
                open_client_order_id=str(open_client_order_id or ""),
            )

            print(f"step {i + 1:>2}: enter {'long' if position_side > 0 else 'short'} at {fill:.2f} qty={qty:.6f}")
            entries += 1
            live_events.append(
                {
                    "step": i + 1,
                    "status": "enter",
                    "direction": int(position_side),
                    "score": float(score),
                    "price": float(fill),
                    "qty": float(qty),
                    "notional": float(notional),
                    "leverage": float(position_leverage),
                    "cash_after_entry": float(cash),
                    "fill_source": fill_source,
                    "meta_label": meta_decision.asdict(),
                    "position_id": current_position_id,
                    "open_client_order_id": current_open_client_order_id,
                }
            )
            cooldown_left = 0

        elif position_side != 0:
            pnl = position_side * (price - entry_price) * qty
            pnl_pct = ((price - entry_price) / entry_price) if position_side > 0 else ((entry_price - price) / entry_price)

            interval_ms = max(1, interval_milliseconds(runtime.interval))
            bars_held = max(
                0,
                (int(latest.timestamp) - int(entry_market_timestamp_ms)) // interval_ms,
            )
            lifecycle_exit = evaluate_position_exit(
                position_side=position_side,
                signal_direction=direction,
                current_pnl_pct=pnl_pct,
                bars_held=bars_held,
                flat_signal_streak=flat_signal_streak,
                stop_loss_pct=cfg.stop_loss_pct,
                take_profit_pct=cfg.take_profit_pct,
                min_position_hold_bars=min_position_hold_bars,
                flat_signal_exit_grace_bars=flat_signal_exit_grace_bars,
                max_position_hold_bars=max_position_hold_bars,
            )
            flat_signal_streak = lifecycle_exit.flat_signal_streak
            should_close = lifecycle_exit.should_close

            if should_close:
                fill = price * (1.0 - position_side * slippage)

                side_to_close = "SELL" if position_side > 0 else "BUY"
                close_position_id = current_position_id
                close_client_order_id = (
                    bot_client_order_id(current_position_id, "close")
                    if not effective_dry_run and current_position_id
                    else None
                )
                try:
                    order_response = _paper_or_live_order(
                        client,
                        runtime,
                        cfg,
                        side=side_to_close,
                        size=abs(qty),
                        dry_run=effective_dry_run,
                        leverage=position_leverage,
                        reduce_only=runtime.market_type == "futures",
                        client_order_id=close_client_order_id,
                    )
                except BinanceAPIError as exc:
                    record_order_error(i + 1, side_to_close, abs(qty), exc)
                    break
                closed_qty, fill, closed_notional, fill_source = _resolved_order_fill_details(
                    client,
                    runtime,
                    order_response,
                    fallback_qty=abs(qty),
                    fallback_price=fill,
                    dry_run=effective_dry_run,
                )
                if closed_qty <= 0.0 or fill <= 0.0:
                    record_order_error(i + 1, side_to_close, abs(qty), BinanceAPIError("Close order response did not include executed quantity"))
                    break
                closed_qty = min(abs(qty), abs(closed_qty))
                realized = position_side * (fill - entry_price) * closed_qty
                exit_fee = abs(closed_notional if closed_notional > 0.0 else fill * closed_qty) * fee_rate
                close_ratio = min(1.0, closed_qty / abs(qty)) if qty else 1.0
                cash += margin_used * close_ratio + realized - exit_fee
                record_signed_live_close(
                    reason=lifecycle_exit.reason,
                    closed_qty=closed_qty,
                    fill_price=fill,
                    realized_pnl=realized,
                    close_ratio=close_ratio,
                    close_fee=exit_fee,
                    order_response=order_response,
                    close_client_order_id=close_client_order_id,
                )
                print(
                    f"step {i + 1:>2}: close {'long' if position_side > 0 else 'short'} "
                    f"pnl={realized:.2f} cash={cash:.2f}"
                )
                closes += 1
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "close",
                        "direction": int(position_side),
                        "score": float(score),
                        "price": float(fill),
                        "pnl": float(realized),
                        "qty_closed": float(closed_qty),
                        "leverage": float(position_leverage),
                        "cash_after": float(cash),
                        "fill_source": fill_source,
                        "bars_held": int(bars_held),
                        "flat_signal_streak": int(flat_signal_streak),
                        "exit_reason": lifecycle_exit.reason,
                        "position_id": close_position_id,
                        "close_client_order_id": str(close_client_order_id or ""),
                    }
                )
                if realized > 0:
                    print("result: win")
                if close_ratio >= 0.999:
                    position_notional = 0.0
                    position_side = 0
                    qty = 0.0
                    margin_used = 0.0
                    entry_price = 0.0
                    entry_market_timestamp_ms = 0
                    position_leverage = leverage
                    flat_signal_streak = 0
                    cooldown_left = max(0, wait_ticks)
                else:
                    qty = max(0.0, abs(qty) - closed_qty)
                    margin_used = max(0.0, margin_used * (1.0 - close_ratio))
                    position_notional = position_side * qty * entry_price
            else:
                unrealized = margin_used + pnl
                print(f"step {i + 1:>2}: hold {'long' if position_side > 0 else 'short'} pnl={pnl_pct:.2%} cash={cash:.2f}")
                print(f"         unrealized equity={cash + unrealized:.2f}")
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "hold",
                        "direction": int(position_side),
                        "score": float(score),
                        "price": float(price),
                        "pnl": float(pnl),
                        "leverage": float(position_leverage),
                        "bars_held": int(bars_held),
                        "flat_signal_streak": int(flat_signal_streak),
                    }
                )

                if direction == 0:
                    cooldown_left = max(0, wait_ticks)

            # safety stop: drop out if drawdown exceeds configured cap
            if position_side != 0:
                equity = cash + margin_used + pnl
            else:
                equity = cash
            if equity > equity_peak:
                equity_peak = equity
            drawdown = (equity_peak - equity) / equity_peak if equity_peak else 0.0
            if drawdown > max_drawdown_seen:
                max_drawdown_seen = drawdown
            if cfg.max_drawdown_limit > 0.0 and drawdown >= cfg.max_drawdown_limit:
                emergency_position_id = ""
                emergency_close_client_order_id: str | None = None
                emergency_leverage = position_leverage
                fill_source = "not_applicable"
                if position_side != 0:
                    fill = price * (1.0 - position_side * slippage)

                    side_to_close = "SELL" if position_side > 0 else "BUY"
                    emergency_position_id = current_position_id
                    emergency_close_client_order_id = (
                        bot_client_order_id(current_position_id, "close")
                        if not effective_dry_run and current_position_id
                        else None
                    )
                    try:
                        order_response = _paper_or_live_order(
                            client,
                            runtime,
                            cfg,
                            side=side_to_close,
                            size=abs(qty),
                            dry_run=effective_dry_run,
                            leverage=position_leverage,
                            reduce_only=runtime.market_type == "futures",
                            client_order_id=emergency_close_client_order_id,
                        )
                    except BinanceAPIError as exc:
                        record_order_error(i + 1, side_to_close, abs(qty), exc)
                        break
                    closed_qty, fill, closed_notional, fill_source = _resolved_order_fill_details(
                        client,
                        runtime,
                        order_response,
                        fallback_qty=abs(qty),
                        fallback_price=fill,
                        dry_run=effective_dry_run,
                    )
                    if closed_qty <= 0.0 or fill <= 0.0:
                        record_order_error(i + 1, side_to_close, abs(qty), BinanceAPIError("Emergency close order response did not include executed quantity"))
                        break
                    closed_qty = min(abs(qty), abs(closed_qty))
                    realized = position_side * (fill - entry_price) * closed_qty
                    exit_fee = abs(closed_notional if closed_notional > 0.0 else fill * closed_qty) * fee_rate
                    close_ratio = min(1.0, closed_qty / abs(qty)) if qty else 1.0
                    cash += margin_used * close_ratio + realized - exit_fee
                    record_signed_live_close(
                        reason="drawdown_limit",
                        closed_qty=closed_qty,
                        fill_price=fill,
                        realized_pnl=realized,
                        close_ratio=close_ratio,
                        close_fee=exit_fee,
                        order_response=order_response,
                        close_client_order_id=emergency_close_client_order_id,
                    )
                    print(
                        f"step {i + 1:>2}: emergency close from drawdown "
                        f"{drawdown:.2%}; cash={cash:.2f}"
                    )
                    if close_ratio >= 0.999:
                        position_notional = 0.0
                        position_side = 0
                        qty = 0.0
                        margin_used = 0.0
                        entry_price = 0.0
                        position_leverage = leverage
                    else:
                        qty = max(0.0, abs(qty) - closed_qty)
                        margin_used = max(0.0, margin_used * (1.0 - close_ratio))
                        position_notional = position_side * qty * entry_price
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "emergency_close",
                        "score": float(score),
                        "drawdown": float(drawdown),
                        "leverage": float(emergency_leverage),
                        "cash_after": float(cash),
                        "fill_source": fill_source,
                        "position_id": emergency_position_id,
                        "close_client_order_id": str(emergency_close_client_order_id or ""),
                    }
                )
                print(f"step {i + 1:>2}: drawdown limit reached ({cfg.max_drawdown_limit:.1%}), stopping loop")
                halt_reason = "drawdown_limit"
                break

        if position_side == 0 and direction != 0:
            live_events.append(
                {
                    "step": i + 1,
                    "status": "signal_no_entry",
                    "score": float(score),
                    "direction": int(direction),
                }
            )

        maybe_grade_sources(i + 1)
        if recovery_observation_active:
            finish_recovery_observation(i + 1, price, "post-interruption-clean-market-read")
            recovery_sleep()
        else:
            live_sleep()
    live_run["result"] = {
        "status": halt_reason,
        "steps_executed": steps_executed,
        "entries": entries,
        "closes": closes,
        "skipped_entries": skipped_entries,
        "model_loads": model_loads,
        "drawdown_seen": float(max_drawdown_seen),
        "ending_cash": float(cash),
        "ending_cash_is_estimate": bool(not effective_dry_run),
        "equity_peak": float(equity_peak),
        "drawdown_limit": drawdown_limit,
    }
    _persist_run_artifact("live", model_path.parent, live_run)
    if max_drawdown_seen > 0.0:
        print(f"max_drawdown observed: {max_drawdown_seen:.2%}")
    cash_label = "cash_estimate" if not effective_dry_run else "cash"
    print(f"finished loop market={runtime.market_type} {cash_label}={cash:.2f}")
    return exit_code


def command_shell(_: argparse.Namespace) -> int:
    from .shell import run_shell

    return int(run_shell([]))


def command_objectives(_: argparse.Namespace) -> int:
    from .objective import describe_objectives

    entries = describe_objectives()
    print(f"{'name':<14} {'label':<14} summary")
    for entry in entries:
        print(f"{entry['name']:<14} {entry['label']:<14} {entry['summary']}")
    return 0


def _load_cli_execution_profile(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    cfg: StrategyConfig,
    *,
    workflow: str,
):
    from .execution_profiles import load_top_of_book_execution_profile

    evidence = load_top_of_book_execution_profile(
        getattr(args, "execution_db", None),
        symbol=runtime.symbol,
        market_type=runtime.market_type,
        strategy=cfg,
    )
    if evidence.warning:
        print(f"{workflow}: {evidence.warning}", file=sys.stderr)
    return evidence


def _print_execution_profile_evidence(evidence: object) -> None:
    profile = getattr(evidence, "profile", None)
    if profile is None:
        if getattr(evidence, "source", "disabled") != "disabled":
            print("execution_profile: unavailable")
        return
    print(
        "execution_profile: "
        f"{getattr(evidence, 'source', 'unknown')} "
        f"spread_bps={float(getattr(evidence, 'spread_bps', profile.spread_bps) or 0.0):.3f} "
        f"depth_notional={float(getattr(evidence, 'depth_notional', profile.quote_volume) or 0.0):.2f} "
        f"liquidity_score={profile.liquidity_score:.3f}"
    )


def command_model_blueprint(args: argparse.Namespace) -> int:
    from .model_blueprint import dumps_blueprint, render_blueprint, validate_blueprint_contract

    errors = validate_blueprint_contract()
    if errors:
        for error in errors:
            print(f"model blueprint invalid: {error}", file=sys.stderr)
        return 2
    include_research = not bool(getattr(args, "implemented_only", False))
    try:
        if getattr(args, "json", False):
            print(dumps_blueprint(risk_level=getattr(args, "risk_level", None), include_research=include_research))
        else:
            print(render_blueprint(risk_level=getattr(args, "risk_level", None), include_research=include_research))
    except ValueError as err:
        print(f"model blueprint failed: {err}", file=sys.stderr)
        return 2
    return 0


def command_train_suite(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    from .objective import get_objective
    from .training_suite import describe_candidate_grid, run_training_suite

    runtime = load_runtime()
    strategy = load_strategy()
    try:
        raw = _load_json_candles(args.input)
    except (OSError, ValueError) as err:
        print(f"failed to load candles from {args.input}: {err}", file=sys.stderr)
        return 2
    candles = []
    for entry in raw:
        try:
            candles.append(Candle(
                open_time=int(entry["open_time"]),
                open=float(entry["open"]),
                high=float(entry["high"]),
                low=float(entry["low"]),
                close=float(entry["close"]),
                volume=float(entry.get("volume", 0.0)),
                close_time=int(entry["close_time"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    try:
        objectives = (
            tuple(get_objective(name).name for name in args.objective)
            if args.objective
            else available_objectives()
        )
        max_candidates = getattr(args, "max_candidates", None)
        if max_candidates is not None and int(max_candidates) < 1:
            raise ValueError("--max-candidates must be >= 1")
        grid_counts = {name: len(describe_candidate_grid(get_objective(name))) for name in objectives}
        if max_candidates is not None:
            grid_counts = {name: min(int(max_candidates), count) for name, count in grid_counts.items()}
        total_candidates = sum(grid_counts.values())
        try:
            backend_label, _backend_info = _workflow_compute_backend(
                runtime,
                getattr(args, "compute_backend", None),
                workflow="training suite",
            )
        except ValueError as exc:
            print(f"training suite failed: {exc}", file=sys.stderr)
            return 2
        print(
            "training suite starting: "
            f"objectives={','.join(objectives)} candidates={total_candidates} "
            f"backend={backend_label} batch_size={getattr(args, 'batch_size', 8192)}",
            flush=True,
        )
        suite_kwargs: dict[str, object] = {
            "objectives": objectives,
            "market_type": runtime.market_type,
            "starting_cash": args.starting_cash,
            "output_dir": Path(args.output_dir),
            "max_workers": args.max_workers,
            "compute_backend": backend_label,
        }
        requested_symbol = str(getattr(args, "symbol", None) or "").strip()
        if requested_symbol:
            training_symbol = normalize_symbol(requested_symbol, default="")
            if not is_supported_major_symbol(training_symbol):
                raise ValueError("train-suite requires a supported BTC, ETH, or SOL symbol")
            suite_kwargs["symbol"] = training_symbol
        else:
            print(
                "training suite warning: --symbol omitted; durable terminal governance is disabled "
                "and produced models are research-only",
                file=sys.stderr,
            )
        if getattr(args, "batch_size", 8192) != 8192:
            suite_kwargs["batch_size"] = max(1, int(args.batch_size))
        if max_candidates is not None:
            suite_kwargs["max_candidates"] = int(max_candidates)
        report = run_training_suite(candles, strategy, **suite_kwargs)
    except ValueError as err:
        print(f"training suite failed: {err}", file=sys.stderr)
        return 2
    print(f"training suite complete: {len(report.outcomes)} objective(s)")
    for outcome in report.outcomes:
        wf_gate = getattr(outcome, "walk_forward_gate", None) or {}
        wf_gate_text = "n/a"
        if wf_gate:
            reason = wf_gate.get("reason")
            wf_gate_text = (
                f"{wf_gate.get('accepted_folds', 0)}/{wf_gate.get('fold_count', 0)}"
                if wf_gate.get("passed")
                else f"failed:{reason or 'fold_gate'}"
            )
        meta = getattr(outcome, "meta_label_report", None) or {}
        meta_text = "n/a"
        if isinstance(meta, dict) and meta:
            meta_status = str(meta.get("status") or "unknown")
            take_count = int(float(meta.get("take_count", 0) or 0))
            downsize_count = int(float(meta.get("downsize_count", 0) or 0))
            skip_count = int(float(meta.get("skip_count", 0) or 0))
            meta_text = f"{meta_status}:take{take_count}/down{downsize_count}/skip{skip_count}"
        selection_risk = getattr(outcome, "selection_risk", None) or {}
        selection_text = "n/a"
        if isinstance(selection_risk, dict) and selection_risk:
            deflated = selection_risk.get("deflated_score")
            penalty = selection_risk.get("trial_penalty")
            trials = selection_risk.get("effective_trials")
            passed = selection_risk.get("passed") is True
            try:
                selection_text = (
                    f"{'pass' if passed else 'fail'}:"
                    f"deflated={float(deflated):+.4f}/penalty={float(penalty):.4f}/trials={int(float(trials))}"
                )
                overfit = selection_risk.get("overfit_diagnostics")
                if isinstance(overfit, dict):
                    pbo = overfit.get("probability_backtest_overfit")
                    status = str(overfit.get("status") or "unknown")
                    if isinstance(pbo, (int, float)) and math.isfinite(float(pbo)):
                        selection_text += f"/pbo={float(pbo):.2f}"
                    elif status == "skipped":
                        selection_text += "/pbo=skipped"
            except (TypeError, ValueError, OverflowError):
                selection_text = "fail:selection_risk_malformed" if not passed else "pass:selection_risk"
        print(
            f"  {outcome.objective:<14} score={outcome.best_score:+.4f} "
            f"threshold={outcome.decision_threshold if outcome.decision_threshold is not None else 'n/a'} "
            f"source={outcome.threshold_source or 'n/a'} "
            f"validation={outcome.validation_score if outcome.validation_score is not None else 'n/a'} "
            f"full={outcome.full_sample_score if outcome.full_sample_score is not None else 'n/a'} "
            f"walk_forward={wf_gate_text} "
            f"selection_risk={selection_text} "
            f"meta_label={meta_text} "
            f"ensemble={'yes' if getattr(outcome, 'ensemble_refined', False) else 'no'} "
            f"hybrid={getattr(outcome, 'hybrid_profile', 'base_only')} "
            f"backend={getattr(outcome, 'training_backend_kind', 'cpu')} "
            f"local_checks={getattr(outcome, 'local_refinement_candidates', 0)} "
            f"ensemble_checks={getattr(outcome, 'ensemble_refinement_candidates', 0)} "
            f"candidates={outcome.explored_candidates} model={outcome.model_path}"
        )
    print(f"summary -> {report.summary_path}")
    return 0


def command_model_lab(args: argparse.Namespace) -> int:
    from .model_lab import run_model_lab
    from .objective import get_objective

    runtime = load_runtime()
    strategy = load_strategy()
    try:
        market_override = getattr(args, "market", None)
        lab_runtime = replace(
            runtime,
            market_type=str(market_override or runtime.market_type).lower(),
            quote_asset=str(getattr(args, "quote_asset", None) or runtime.quote_asset).upper(),
            interval=str(getattr(args, "interval", None) or runtime.interval),
        )
        objectives = (
            tuple(get_objective(name).name for name in args.objective)
            if args.objective
            else available_objectives()
        )
        if int(args.max_symbols) < 1:
            raise ValueError("--max-symbols must be >= 1")
        if int(args.max_scan) < 1:
            raise ValueError("--max-scan must be >= 1")
        if int(args.limit) < 100:
            raise ValueError("--limit must be >= 100")
        max_candidates = getattr(args, "max_candidates", None)
        if max_candidates is not None and int(max_candidates) < 1:
            raise ValueError("--max-candidates must be >= 1")
        backend_label, _backend_info = _workflow_compute_backend(
            lab_runtime,
            getattr(args, "compute_backend", None),
            workflow="model lab",
        )
        report = run_model_lab(
            _build_client(lab_runtime),
            lab_runtime,
            strategy,
            objectives=objectives,
            output_dir=Path(args.output_dir),
            starting_cash=float(args.starting_cash),
            max_symbols=int(args.max_symbols),
            max_scan=int(args.max_scan),
            limit=int(args.limit),
            compute_backend=backend_label,
            batch_size=max(1, int(args.batch_size)),
            score_batch_size=(
                max(1, int(args.score_batch_size))
                if getattr(args, "score_batch_size", None) is not None
                else None
            ),
            max_candidates=(int(max_candidates) if max_candidates is not None else None),
            learning_feedback_path=(
                Path(args.learning_feedback)
                if getattr(args, "learning_feedback", None)
                else None
            ),
            full_history=bool(getattr(args, "full_history", False)),
            market_db_path=(
                Path(getattr(args, "market_db", None))
                if getattr(args, "market_db", None)
                else (Path("data/market_data.sqlite") if getattr(args, "require_db_data", False) else None)
            ),
            require_db_data=bool(getattr(args, "require_db_data", False)),
        )
    except (BinanceAPIError, ValueError) as exc:
        print(f"model lab failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"model lab complete: accepted={len(report.accepted_symbols)}/{len(report.outcomes)} "
        f"symbols market={report.market_type} interval={getattr(report, 'interval', lab_runtime.interval)} "
        f"source={getattr(report, 'data_source', 'unknown')} objectives={','.join(objectives)}"
    )
    portfolio_risk = getattr(report, "portfolio_risk", None) or {}
    if portfolio_risk:
        status = "pass" if portfolio_risk.get("accepted") else "fail"
        reason = str(portfolio_risk.get("reason") or "ok")
        print(
            "portfolio risk: "
            f"{status} deployed={float(portfolio_risk.get('deployed_weight', 0.0)):.1%} "
            f"effective_symbols={float(portfolio_risk.get('effective_symbol_count', 0.0)):.2f} "
            f"corr_effective={float(portfolio_risk.get('correlation_adjusted_effective_symbol_count', 0.0)):.2f} "
            f"cvar95={float(portfolio_risk.get('portfolio_cvar_95', 0.0)):.2%} "
            f"max_corr={float(portfolio_risk.get('max_pairwise_correlation', 0.0)):.2f} "
            f"reason={reason}"
        )
    for outcome in report.outcomes:
        status = "accepted" if outcome.accepted else "rejected"
        score_text = ", ".join(f"{key}={value:+.4f}" for key, value in sorted(outcome.objective_scores.items()))
        hybrid_text = ", ".join(f"{key}:{value}" for key, value in sorted(outcome.hybrid_profiles.items()))
        stress = outcome.stress_validation or {}
        stress_text = "n/a"
        if stress:
            stress_text = (
                "pass"
                if stress.get("accepted")
                else (
                    f"fail worst_pnl={float(stress.get('worst_realized_pnl', 0.0)):+.2f} "
                    f"worst_dd={float(stress.get('worst_max_drawdown', 0.0)):.2%}"
                )
            )
        robustness = getattr(outcome, "robustness_validation", None) or {}
        robustness_text = "n/a"
        if robustness:
            edge_p = robustness.get("worst_sign_test_p_value")
            edge_lower = robustness.get("worst_bootstrap_lower_mean_return")
            edge_text = ""
            if isinstance(edge_p, (int, float)) and isinstance(edge_lower, (int, float)):
                edge_text = f" edge_p={float(edge_p):.3f} lower={float(edge_lower):+.2%}"
            robustness_text = (
                f"pass {int(robustness.get('accepted_windows', 0))}/{int(robustness.get('window_count', 0))}{edge_text}"
                if robustness.get("accepted")
                else (
                    f"fail {int(robustness.get('accepted_windows', 0))}/{int(robustness.get('window_count', 0))} "
                    f"worst_pnl={float(robustness.get('worst_realized_pnl', 0.0)):+.2f}{edge_text}"
                )
            )
        detail = score_text or outcome.error or "no accepted objectives"
        coverage = getattr(outcome, "data_coverage", None) or {}
        coverage_text = "coverage=n/a"
        if isinstance(coverage, dict):
            coverage_text = (
                f"span={coverage.get('used_start_utc') or 'n/a'}->{coverage.get('used_end_utc') or 'n/a'} "
                f"years={float(coverage.get('used_duration_years') or 0.0):.2f} "
                f"candles={int(coverage.get('candles_used') or 0)} "
                f"coverage={float(coverage.get('coverage_ratio') or 0.0):.4f} "
                f"integrity={coverage.get('integrity_status') or 'unknown'} "
                f"source={coverage.get('source_scope') or 'unknown'}"
            )
        print(
            f"  {status:<8} {outcome.symbol:<12} rows={outcome.rows:<5} {detail} "
            f"hybrid={hybrid_text or 'n/a'} stress={stress_text} robustness={robustness_text} "
            f"{coverage_text}"
        )
    print(f"summary -> {report.report_path}")
    return 0 if report.accepted_symbols else 2


def command_backtest_panel(args: argparse.Namespace) -> int:
    from .backtest_panel import BacktestRequest, parse_date_ms, run_panel

    runtime = load_runtime()
    strategy = load_strategy()
    market = args.market or runtime.market_type
    try:
        compute_backend, _backend_info = _workflow_compute_backend(
            runtime,
            getattr(args, "compute_backend", None),
            workflow="backtest panel",
        )
        request = BacktestRequest(
            interval=args.interval,
            market_type=market,
            symbol=runtime.symbol,
            start_ms=parse_date_ms(args.from_date),
            end_ms=parse_date_ms(args.to_date, end_of_day=True),
            model_path=args.model,
            data_path=args.input,
            execution_db=getattr(args, "execution_db", None),
            compute_backend=compute_backend,
            starting_cash=args.starting_cash,
            objective=args.objective,
            tag=args.tag,
            notes=args.notes,
        )
        report = run_panel(request, strategy)
    except ValueError as err:
        print(f"backtest-panel: {err}", file=sys.stderr)
        return 2
    print(f"backtest report -> data/backtests/{report.filename}")
    coverage = getattr(report, "data_coverage", None)
    if coverage is not None:
        print(
            f"  data_span={coverage.used_start_utc or 'n/a'}->{coverage.used_end_utc or 'n/a'} "
            f"interval={coverage.interval} candles={coverage.candles_used}/{coverage.candles_available} "
            f"rows={coverage.rows_used} years={coverage.used_duration_years:.2f} "
            f"coverage={coverage.coverage_ratio:.4f} gaps={coverage.gap_count}"
        )
    print(
        f"  trades={report.result.closed_trades} "
        f"realized_pnl={report.result.realized_pnl:+.2f} "
        f"max_dd={report.result.max_drawdown:.2%}"
    )
    if report.objective_score is not None:
        print(f"  objective={args.objective} score={report.objective_score:+.4f}")
    return 0


def _build_autonomous_decision_fn(
    *,
    model_path: Path,
    strategy: StrategyConfig,
    effective_dry_run: bool,
    require_model_candidate_search: bool = False,
    require_accelerator_evidence: bool = False,
    require_live_data_evidence: bool = False,
    require_microstructure_evidence: bool = False,
    expected_symbol: str | None = None,
    expected_market_type: str | None = None,
    expected_interval: str | None = None,
    minimum_model_taker_fee_bps: float | None = None,
):
    from .autonomous import Decision

    model, model_error, model_notice = _load_live_start_model(
        model_path,
        strategy,
        effective_dry_run=effective_dry_run,
        require_model_candidate_search=require_model_candidate_search,
        require_accelerator_evidence=require_accelerator_evidence,
        require_live_data_evidence=require_live_data_evidence,
        require_microstructure_evidence=require_microstructure_evidence,
        expected_symbol=expected_symbol,
        expected_market_type=expected_market_type,
        expected_interval=expected_interval,
    )
    if model_error is not None:
        return None, model_error, model_notice
    if model is None and not effective_dry_run:
        return None, f"Autonomous live mode requires a compatible model: {model_path}", model_notice
    if not effective_dry_run and minimum_model_taker_fee_bps is not None:
        fee_error = _model_taker_fee_compatibility_error(model, minimum_model_taker_fee_bps)
        if fee_error is not None:
            return None, f"Autonomous startup blocked: {fee_error}", model_notice
    effective_strategy = apply_model_strategy_overrides(strategy, model) if model is not None else strategy
    state: dict[str, object] = {"model": model, "step": 0}

    def decide(client, runtime: RuntimeConfig, current_strategy: StrategyConfig, _objective) -> Decision:
        state["step"] = int(state["step"]) + 1
        current_model = cast(TrainedModel | None, state["model"])
        candles = client.get_klines(runtime.symbol, runtime.interval, limit=max(current_strategy.model_lookback, 300))
        rows = _live_rows_for_model(candles, current_strategy, current_model, compute_backend=runtime.compute_backend)
        training_rows = (
            _readiness_model_rows(candles, current_strategy, current_model, compute_backend=runtime.compute_backend)
            if current_model is not None
            else _build_model_rows(candles, current_strategy, compute_backend=runtime.compute_backend)
        )
        if not rows:
            price, observed_at_ms = client.get_symbol_price(runtime.symbol)
            return Decision(
                side="FLAT",
                confidence=0.0,
                mark_price=float(price),
                observed_at_ms=int(observed_at_ms),
            )
        if current_model is None:
            if not effective_dry_run:  # pragma: no cover - rejected before the decision function is returned
                raise RuntimeError(f"Autonomous live mode requires a compatible model: {model_path}")
            if not training_rows:
                price, observed_at_ms = client.get_symbol_price(runtime.symbol)
                return Decision(
                    side="FLAT",
                    confidence=0.0,
                    mark_price=float(price),
                    observed_at_ms=int(observed_at_ms),
                )
            current_model = train(
                training_rows,
                epochs=40,
                feature_signature=_live_model_feature_signature(None, current_strategy),
                compute_backend=runtime.compute_backend,
                batch_size=8192,
            )
            state["model"] = current_model
        latest = rows[-1]
        raw_score = current_model.predict_proba(latest.features)
        threshold = model_decision_threshold(current_model, current_strategy.signal_threshold)
        score = confidence_adjusted_probability(raw_score, current_strategy.confidence_beta)
        decision_strategy = current_strategy
        if current_strategy.external_signals_enabled:
            external_report = collect_external_signals(
                symbol=runtime.symbol,
                cache_path=_external_signal_cache_path(model_path),
                ttl_seconds=current_strategy.external_signal_ttl_seconds,
                timeout_seconds=current_strategy.external_signal_timeout_seconds,
                max_adjustment=current_strategy.external_signal_max_adjustment,
                min_providers=current_strategy.external_signal_min_providers,
                compute_backend=runtime.compute_backend,
                short_reaction_refresh_seconds=current_strategy.external_signal_short_reaction_refresh_seconds,
                news_provider_limit=current_strategy.external_signal_news_provider_limit,
                news_items_per_provider=current_strategy.external_signal_news_items_per_provider,
                news_provider_parallelism=current_strategy.external_signal_provider_parallelism,
                news_provider_jitter_seconds=current_strategy.external_signal_provider_jitter_seconds,
                ollama_news_enabled=current_strategy.external_news_ai_enabled,
                ollama_model=current_strategy.external_news_ai_model,
                ollama_url=current_strategy.external_news_ai_url,
                ollama_timeout_seconds=current_strategy.external_news_ai_timeout_seconds,
                telemetry_path=current_strategy.telemetry_db_path if current_strategy.telemetry_enabled else None,
                source_grade_max_age_hours=current_strategy.source_grade_max_age_hours,
            )
            score, decision_strategy, _applied_adjustment = _apply_external_signal_to_score(score, current_strategy, external_report)
        regime_window_size = max(8, min(len(rows), int(decision_strategy.liquidity_lookback_bars)))
        regime_evidence = classify_market_regime(rows[-regime_window_size:])
        regime_score = market_regime_unpredictability(
            regime_evidence.dominant_regime,
            regime_evidence.confidence,
            regime_evidence.notes,
        )
        base_long_threshold, base_short_threshold = model_direction_thresholds(
            current_model,
            decision_strategy.signal_threshold,
            market_type=runtime.market_type,
        )
        side_thresholds_explicit = (
            getattr(current_model, "long_decision_threshold", None) is not None
            or getattr(current_model, "short_decision_threshold", None) is not None
        )
        base_threshold = _directional_threshold_confidence(base_long_threshold, base_short_threshold, threshold)
        liquidity_adjustment = liquidity_session_adjustment(rows, len(rows) - 1, decision_strategy, base_threshold)
        threshold_add = float(liquidity_adjustment.threshold) - float(base_threshold)
        effective_long_threshold, effective_short_threshold = _adjust_directional_thresholds(
            base_long_threshold,
            base_short_threshold,
            threshold_add,
        )
        effective_threshold = _directional_threshold_confidence(
            effective_long_threshold,
            effective_short_threshold,
            threshold,
        )
        _record_model_telemetry(
            current_strategy,
            step=int(state["step"]),
            row=latest,
            raw_score=raw_score,
            adjusted_score=score,
            threshold=effective_threshold,
            model=current_model,
            runtime=runtime,
        )
        direction = _score_to_direction(
            score,
            decision_strategy,
            runtime.market_type,
            effective_long_threshold,
            short_threshold=effective_short_threshold,
            side_thresholds_explicit=side_thresholds_explicit,
        )
        meta_threshold = (
            1.0 - effective_short_threshold
            if runtime.market_type == "futures" and direction < 0 and effective_short_threshold is not None
            else effective_threshold
        )
        base_meta_decision = apply_meta_label_policy(
            getattr(current_model, "meta_label_policy", {}),
            adjusted_probability=score,
            threshold=meta_threshold,
            side=direction,
            market_type=runtime.market_type,
        )
        meta_decision = apply_liquidity_session_meta(base_meta_decision, liquidity_adjustment) if direction != 0 else base_meta_decision
        if direction > 0:
            side = "LONG"
        elif direction < 0:
            side = "SHORT"
        else:
            side = "FLAT"
        return Decision(
            side=side,
            confidence=float(score),
            mark_price=float(latest.close),
            size_multiplier=float(meta_decision.size_multiplier),
            meta_label_action=str(meta_decision.action),
            meta_label_reason=str(meta_decision.reason),
            regime=regime_evidence.dominant_regime,
            regime_confidence=float(regime_evidence.confidence),
            regime_notes=tuple(regime_evidence.notes),
            regime_unpredictability_score=float(regime_score),
            observed_at_ms=int(latest.timestamp),
        )

    # Carry the model's tested execution contract into the coordinator without
    # breaking the public three-value return shape used by integrations.
    setattr(decide, "_effective_strategy", effective_strategy)
    return decide, None, model_notice


def command_autonomous(args: argparse.Namespace) -> int:
    from .autonomous import (
        AutonomousConfig,
        STATE_PAUSED,
        STATE_RUNNING,
        STATE_STOPPING,
        AutonomousControl,
        close_tracked_open_positions,
        run_loop,
    )
    from .objective import get_objective
    from .positions import PositionsStore

    control = AutonomousControl()
    action = args.action
    if action == "start":
        try:
            objective = get_objective(args.objective)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        runtime = load_runtime()
        strategy = _strategy_with_profile(load_strategy(), objective.name)
        if getattr(args, "live", False):
            effective_dry_run = False
        elif getattr(args, "paper", False):
            effective_dry_run = True
        else:
            effective_dry_run = bool(runtime.dry_run)
        if not _allows_signed_execution(runtime):
            print("Autonomous mode requires testnet=true or demo=true.", file=sys.stderr)
            return 2
        if not effective_dry_run and not _has_api_credentials(runtime):
            print(_credential_required_message("Autonomous live mode"), file=sys.stderr)
            return 2
        model_path = Path(getattr(args, "model", "data/model.json"))
        try:
            strategy, commission_assumption = apply_offline_commission_floor(
                strategy,
                market_type=runtime.market_type,
                symbol=runtime.symbol,
            )
        except ValueError as exc:
            print(f"Autonomous startup blocked: commission floor failed: {exc}", file=sys.stderr)
            return 2
        decision_fn, model_error, model_notice = _build_autonomous_decision_fn(
            model_path=model_path,
            strategy=strategy,
            effective_dry_run=effective_dry_run,
            require_model_candidate_search=not effective_dry_run,
            require_live_data_evidence=not effective_dry_run,
            require_microstructure_evidence=not effective_dry_run and runtime.market_type == "futures",
            expected_symbol=runtime.symbol,
            expected_market_type=runtime.market_type,
            expected_interval=runtime.interval,
            minimum_model_taker_fee_bps=commission_assumption.modeled_taker_fee_bps,
        )
        if model_error is not None or decision_fn is None:
            print(model_error or f"Autonomous mode requires a readable model: {model_path}", file=sys.stderr)
            return 2
        try:
            client = _build_client(runtime)
            if not effective_dry_run:
                verified_strategy, verified_assumption, _commission_rates = apply_verified_commission_rate(
                    strategy,
                    client=client,
                    symbol=runtime.symbol,
                )
                if (
                    verified_assumption.modeled_taker_fee_bps
                    > commission_assumption.modeled_taker_fee_bps
                ):
                    decision_fn, model_error, model_notice = _build_autonomous_decision_fn(
                        model_path=model_path,
                        strategy=verified_strategy,
                        effective_dry_run=effective_dry_run,
                        require_model_candidate_search=True,
                        require_live_data_evidence=True,
                        require_microstructure_evidence=runtime.market_type == "futures",
                        expected_symbol=runtime.symbol,
                        expected_market_type=runtime.market_type,
                        expected_interval=runtime.interval,
                        minimum_model_taker_fee_bps=(
                            verified_assumption.modeled_taker_fee_bps
                        ),
                    )
                    if model_error is not None or decision_fn is None:
                        print(
                            model_error
                            or "Autonomous startup blocked after verified commission update.",
                            file=sys.stderr,
                        )
                        return 2
                strategy = verified_strategy
                commission_assumption = verified_assumption
        except (BinanceAPIError, ValueError) as exc:
            print(
                f"Autonomous startup blocked: commission-rate verification failed: {exc}",
                file=sys.stderr,
            )
            return 2
        if model_notice is not None:
            print(model_notice, file=sys.stderr)
        model_strategy = getattr(decision_fn, "_effective_strategy", None)
        if isinstance(model_strategy, StrategyConfig):
            strategy = model_strategy
        cfg = AutonomousConfig(
            objective=objective.name,
            poll_seconds=max(0.0, float(getattr(args, "poll_seconds", 30.0))),
            stop_after_iterations=getattr(args, "iterations", None),
            heartbeat_every=max(1, int(getattr(args, "heartbeat_every", 1))),
            dry_run=effective_dry_run,
            starting_reference_cash=max(0.0, float(getattr(args, "starting_cash", 1000.0))),
        )
        print(
            "taker fee: "
            f"{commission_assumption.modeled_taker_fee_bps:.4f} bps "
            f"source={commission_assumption.source} verified={str(commission_assumption.verified).lower()}"
        )
        result = run_loop(
            client,
            runtime,
            strategy,
            cfg,
            decision_fn=decision_fn,
        )
        print(
            "autonomous: "
            f"{result.exit_reason} iterations={result.iterations} "
            f"opened={result.opened_trades} closed={result.closed_trades} skipped={result.skipped_entries}"
        )
        return 0
    if action == "pause":
        control.write(STATE_PAUSED, note="CLI pause")
        print("autonomous: PAUSED")
        return 0
    if action == "resume":
        control.write(STATE_RUNNING, note="CLI resume")
        print("autonomous: RUNNING")
        return 0
    if action == "stop":
        control.write(STATE_STOPPING, note="CLI stop")
        runtime = load_runtime()
        strategy = load_strategy()
        mark_price: float | None = None
        client: BinanceClient | None = None
        try:
            client = _build_client(runtime)
            quote_fn = getattr(client, "get_symbol_price", None)
            if not callable(quote_fn):
                raise AttributeError("client does not expose get_symbol_price")
            price, _ts = quote_fn(runtime.symbol)
            mark_price = float(price)
        except (AttributeError, BinanceAPIError, TypeError, ValueError) as exc:
            print(f"autonomous: stop quote unavailable; local positions will close at entry price if needed ({exc})", file=sys.stderr)
        close_report = close_tracked_open_positions(
            PositionsStore(),
            mark_price,
            "operator-stop-command",
            client=client,
            reduce_only=runtime.market_type == "futures" and strategy.reduce_only_on_close,
        )
        suffix = f"; closed_positions={close_report.closed}" if close_report.closed else ""
        if close_report.skipped or close_report.failed:
            failures = ", ".join(close_report.failures[:5])
            print(
                f"autonomous: STOPPING{suffix}; close_incomplete "
                f"skipped={close_report.skipped} failed={close_report.failed} {failures}",
                file=sys.stderr,
            )
            return 2
        print(f"autonomous: STOPPING{suffix}")
        return 0
    # status
    payload = control.read()
    print(
        f"state={payload.get('state')} note={payload.get('note') or ''} "
        f"ts_ms={payload.get('ts_ms')}"
    )
    return 0


def command_positions(args: argparse.Namespace) -> int:
    from .positions import (
        PositionsStore,
        compute_stats,
        load_learning_feedback,
        render_positions_table,
        render_learning_feedback,
        render_stats_lines,
    )

    store = PositionsStore()
    opens = store.load_open()
    if not opens:
        print("(no open positions)")
    else:
        for row in render_positions_table(opens, mark_price=None):
            print(row)
    if args.stats:
        stats = compute_stats(store, mark_price=None)
        print("")
        for line in render_stats_lines(stats):
            print(line)
    if getattr(args, "learning", False):
        report = load_learning_feedback(store)
        print("")
        for line in render_learning_feedback(report):
            print(line)
    return 0


def command_close(args: argparse.Namespace) -> int:
    from .positions import PositionsStore

    store = PositionsStore()
    target = args.position_id
    if target.lower() == "all":
        opens = store.load_open()
        if not opens:
            print("no open positions")
            return 0
        print(
            "refusing local-ledger erasure; use autonomous stop so every "
            "bot-owned position is closed and reconciled through its venue",
            file=sys.stderr,
        )
        return 2
    position = store.find_open(target)
    if position is None:
        print(f"no open position with id {target!r}", file=sys.stderr)
        return 1
    print(
        f"refusing local-ledger erasure for position {target}; use autonomous "
        "stop so the venue close and ownership reconciliation remain atomic",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return command_menu(argparse.Namespace())
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
