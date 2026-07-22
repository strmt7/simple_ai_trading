"""Exact, return-independent implementation contract for Round 72."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Sequence


ROUND72_IMPLEMENTATION_SCHEMA = "round-072-price-discovery-implementation-v2"
ROUND72_IMPLEMENTATION_V1_SHA256 = (
    "17428ce4109f7ffc827ea8b9aa545bc16a5aebb2bd05466de0ca93d540891fbd"
)
WINDOWS_SECONDS = (1, 5, 15, 30, 60, 300, 900)
HORIZONS_SECONDS = (30, 60, 300)
ENTRY_DELAYS_SECONDS = (2, 5)
PRIMARY_ENTRY_DELAY_SECONDS = 2
STRESS_ENTRY_DELAY_SECONDS = 5
ANCHOR_SPACING_SECONDS = 30
ANCHOR_SECOND_OFFSET = 29
MAXIMUM_FEATURE_WINDOW_SECONDS = 900
FEATURE_BURN_IN_SECONDS = 1_800
FEATURE_LAYERS = ("perpetual_only", "spot_perpetual", "cross_asset")
PRIMARY_LOSS_METRICS = {
    "binary_direction": ("log_loss", "brier_score"),
    "continuous_return_bps": ("mean_squared_error", "mean_absolute_error"),
}
MARKET_WINDOW_METRICS = (
    "log_return_bps",
    "path_variation_bps",
    "realized_volatility_bps",
    "log_range_bps",
    "signed_quote_flow",
    "log1p_quote_volume",
    "aggregate_orders_per_second",
    "log1p_mean_aggregate_quote",
    "log1p_maximum_aggregate_quote",
    "aggregate_quote_hhi",
    "log1p_constituent_per_aggregate",
    "zero_flow_fraction",
)
MARKET_CHANGE_METRICS = (
    "realized_volatility_bps",
    "signed_quote_flow",
    "log1p_quote_volume",
    "aggregate_orders_per_second",
)
PAIR_WINDOW_METRICS = (
    "basis_change_bps",
    "spot_minus_perpetual_return_bps",
    "signed_flow_product",
    "absolute_signed_flow_difference",
    "log_relative_quote_volume",
    "log_relative_aggregate_rate",
)
SPOT_FLOW_LAGS_SECONDS = (1, 5, 15, 30)
CROSS_ASSET_WINDOW_METRICS = (
    "perpetual_return_mean_bps",
    "perpetual_return_dispersion_bps",
    "perpetual_return_directional_agreement",
    "perpetual_signed_flow_mean",
    "perpetual_signed_flow_dispersion",
    "leader_perpetual_return_bps",
    "leader_perpetual_signed_flow",
)
CLOCK_FEATURE_NAMES = (
    "utc_day_phase_sine",
    "utc_day_phase_cosine",
    "utc_weekday_phase_sine",
    "utc_weekday_phase_cosine",
    "utc_minute_phase_sine",
    "utc_minute_phase_cosine",
    "utc_five_minute_phase_sine",
    "utc_five_minute_phase_cosine",
    "utc_fifteen_minute_phase_sine",
    "utc_fifteen_minute_phase_cosine",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _canonical_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("frozen_at_utc must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def market_feature_names(market: str) -> tuple[str, ...]:
    prefix = str(market).strip().lower()
    if prefix not in {"spot", "perpetual"}:
        raise ValueError("market feature prefix must be spot or perpetual")
    names = [f"{prefix}_last_trade_age_log1p"]
    for window in WINDOWS_SECONDS:
        names.extend(
            f"{prefix}_{metric}_{window}s" for metric in MARKET_WINDOW_METRICS
        )
        names.extend(
            f"{prefix}_{metric}_change_{window}s"
            for metric in MARKET_CHANGE_METRICS
        )
    return tuple(names)


def pair_feature_names() -> tuple[str, ...]:
    names = ["perpetual_minus_spot_basis_bps"]
    for window in WINDOWS_SECONDS:
        names.extend(f"{metric}_{window}s" for metric in PAIR_WINDOW_METRICS)
    for lag in SPOT_FLOW_LAGS_SECONDS:
        names.extend(
            (
                f"spot_lagged_flow_perpetual_flow_product_{lag}s",
                f"spot_lagged_flow_minus_perpetual_flow_{lag}s",
            )
        )
    return tuple(names)


def cross_asset_feature_names() -> tuple[str, ...]:
    return tuple(
        f"cross_asset_{metric}_{window}s"
        for window in WINDOWS_SECONDS
        for metric in CROSS_ASSET_WINDOW_METRICS
    )


def layer_feature_names(layer: str) -> tuple[str, ...]:
    value = str(layer).strip().lower()
    if value not in FEATURE_LAYERS:
        raise ValueError(f"unknown Round 72 feature layer: {layer}")
    names = (*market_feature_names("perpetual"), *CLOCK_FEATURE_NAMES)
    if value in {"spot_perpetual", "cross_asset"}:
        names = (*names, *market_feature_names("spot"), *pair_feature_names())
    if value == "cross_asset":
        names = (*names, *cross_asset_feature_names())
    if len(names) != len(set(names)):
        raise RuntimeError("Round 72 feature names are not unique")
    return tuple(names)


def _rolling_folds() -> list[dict[str, object]]:
    # Month positions are frozen against the 2020-10..2026-03 development panel.
    months = []
    year, month = 2020, 10
    while (year, month) <= (2026, 3):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    folds = []
    for index in range(6):
        training_end = 24 + index * 6
        tuning_end = training_end + 6
        test_end = tuning_end + 6
        folds.append(
            {
                "fold": index + 1,
                "training_months": [months[0], months[training_end - 1]],
                "training_month_count": training_end,
                "tuning_months": [months[training_end], months[tuning_end - 1]],
                "test_months": [months[tuning_end], months[test_end - 1]],
            }
        )
    if folds[-1]["test_months"][-1] != "2026-03":
        raise RuntimeError("Round 72 rolling fold construction drifted")
    return folds


def build_round72_implementation_spec(
    *,
    design_sha256: str,
    inventory_sha256: str,
    inventory_file_sha256: str,
    frozen_at_utc: str,
) -> dict[str, object]:
    """Build the full implementation freeze without reading prices or returns."""

    perpetual_names = layer_feature_names("perpetual_only")
    paired_names = layer_feature_names("spot_perpetual")
    cross_names = layer_feature_names("cross_asset")
    without_hash: dict[str, object] = {
        "schema_version": ROUND72_IMPLEMENTATION_SCHEMA,
        "round": 72,
        "frozen_at_utc": _canonical_utc(frozen_at_utc),
        "amendment": {
            "predecessor_implementation_sha256": ROUND72_IMPLEMENTATION_V1_SHA256,
            "reason": "pre-result clarification of already-declared design metrics, aggregation, resampling, stress, seed, and continuous-market semantics",
            "round72_price_or_return_result_available_before_amendment": False,
            "round72_model_result_available_before_amendment": False,
            "data_feature_target_split_or_model_parameter_changed": False,
        },
        "design_sha256": _require_sha256(design_sha256, "design_sha256"),
        "inventory_sha256": _require_sha256(inventory_sha256, "inventory_sha256"),
        "inventory_file_sha256": _require_sha256(
            inventory_file_sha256, "inventory_file_sha256"
        ),
        "freeze_evidence": {
            "price_or_return_values_read_to_choose_formulas": False,
            "model_result_available_at_freeze": False,
            "single_smallest_archive_schema_smoke_preceded_this_freeze": True,
            "schema_smoke_used_for_feature_or_parameter_selection": False,
            "post_result_changes_permitted": False,
        },
        "anchor_contract": {
            "calendar": "UTC",
            "market_session_semantics": "Binance spot and USD-M perpetual markets are continuous; a UTC day is only a sampling and resampling block, never a formal market close",
            "listed_product_session_semantics": "ETF and listed-futures sessions are separate timestamped external context and are excluded from Round 72 features",
            "anchor_spacing_seconds": ANCHOR_SPACING_SECONDS,
            "anchor_second_offset_within_spacing": ANCHOR_SECOND_OFFSET,
            "anchor_rule": "second_ms modulo 30000 equals 29000; available_time_ms is second_ms plus 1000",
            "maximum_window_seconds": MAXIMUM_FEATURE_WINDOW_SECONDS,
            "burn_in_seconds": FEATURE_BURN_IN_SECONDS,
            "window_rule": "current window includes anchor and preceding w-1 seconds; comparison window is the immediately preceding disjoint w seconds",
            "day_boundary_windows_permitted": False,
            "target_symbol_spot_age_maximum_seconds": 2,
            "target_symbol_perpetual_age_maximum_seconds": 2,
            "peer_age_gate_for_optional_cross_asset_features": False,
        },
        "feature_contract": {
            "windows_seconds": list(WINDOWS_SECONDS),
            "market_window_metrics": list(MARKET_WINDOW_METRICS),
            "market_change_metrics": list(MARKET_CHANGE_METRICS),
            "pair_window_metrics": list(PAIR_WINDOW_METRICS),
            "spot_flow_lags_seconds": list(SPOT_FLOW_LAGS_SECONDS),
            "cross_asset_window_metrics": list(CROSS_ASSET_WINDOW_METRICS),
            "clock_features": list(CLOCK_FEATURE_NAMES),
            "formulas": {
                "log_return_bps": "10000*log(close_t/close_t_minus_w)",
                "path_variation_bps": "10000*sum(abs(one_second_log_return)) over the window",
                "realized_volatility_bps": "10000*sqrt(sum(one_second_log_return_squared)) over the window",
                "log_range_bps": "10000*log(max(high)/min(low)) over the window",
                "signed_quote_flow": "sum(aggressive_buy_quote-aggressive_sell_quote)/sum(quote_volume), zero when denominator is zero",
                "log1p_quote_volume": "log1p(sum(quote_volume))",
                "aggregate_orders_per_second": "sum(aggregate_count)/window_seconds",
                "log1p_mean_aggregate_quote": "log1p(sum(quote_volume)/sum(aggregate_count)), zero when denominator is zero",
                "log1p_maximum_aggregate_quote": "log1p(max(maximum_aggregate_quote))",
                "aggregate_quote_hhi": "sum(squared_aggregate_quote_sum)/sum(quote_volume)^2, zero when denominator is zero",
                "log1p_constituent_per_aggregate": "log1p(sum(constituent_trade_count)/sum(aggregate_count)), zero when denominator is zero",
                "zero_flow_fraction": "mean(aggregate_count==0)",
                "change": "current metric minus the same metric over the immediately preceding disjoint window",
                "basis_bps": "10000*log(perpetual_close/spot_close)",
                "basis_change_bps": "basis_t minus basis_t_minus_w",
                "signed_flow_product": "spot_signed_quote_flow*perpetual_signed_quote_flow for the same window",
                "absolute_signed_flow_difference": "abs(spot_signed_quote_flow-perpetual_signed_quote_flow)",
                "log_relative_quote_volume": "spot_log1p_quote_volume-perpetual_log1p_quote_volume",
                "log_relative_aggregate_rate": "log1p(spot aggregate rate)-log1p(perpetual aggregate rate)",
                "leader_definition": "BTC for ETH and SOL; equal-weight ETH/SOL mean for BTC",
            },
            "layer_feature_counts": {
                "perpetual_only": len(perpetual_names),
                "spot_perpetual": len(paired_names),
                "cross_asset": len(cross_names),
            },
            "layer_feature_names_sha256": {
                "perpetual_only": _canonical_sha256(perpetual_names),
                "spot_perpetual": _canonical_sha256(paired_names),
                "cross_asset": _canonical_sha256(cross_names),
            },
            "nonfinite_policy": "exclude the anchor before any role assignment and report by symbol and month",
            "normalization": "none for LightGBM",
        },
        "target_contract": {
            "horizons_seconds": list(HORIZONS_SECONDS),
            "entry_delays_seconds": list(ENTRY_DELAYS_SECONDS),
            "primary_entry_delay_seconds": PRIMARY_ENTRY_DELAY_SECONDS,
            "stress_entry_delay_seconds": STRESS_ENTRY_DELAY_SECONDS,
            "entry_index": "anchor_index plus entry_delay_seconds",
            "exit_index": "anchor_index plus entry_delay_seconds plus horizon_seconds",
            "transaction_vwap": "quote_volume/base_volume in the exact second",
            "eligibility": "both entry and exit perpetual seconds must contain at least one aggregate trade and positive base volume",
            "continuous_bps": "10000*log(exit_vwap/entry_vwap)",
            "binary_label": "one only when continuous_bps is strictly positive; exact zero belongs to class zero",
            "primary_models_are_not_refit_for_stress_delay": True,
            "profit_or_executable_fill_target": False,
        },
        "split_contract": {
            "development_months": ["2020-10", "2026-03"],
            "terminal_holdout_months_never_read": ["2026-04", "2026-05", "2026-06"],
            "folds": _rolling_folds(),
            "purge": "every training or tuning label exit must precede the first anchor available time of the following role",
            "symbol_pooling": False,
            "horizon_pooling": False,
        },
        "model_contract": {
            "feature_layers": list(FEATURE_LAYERS),
            "optional_cross_asset_layer_runs_only_after_primary_spot_perpetual_gate_passes": True,
            "heads": ["binary_direction", "continuous_return_bps"],
            "parameters": {
                "boosting": "gbdt",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 4,
                "min_data_in_leaf": 500,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l1": 1.0,
                "lambda_l2": 1.0,
                "max_bin": 63,
                "maximum_boosting_iterations": 256,
                "early_stopping_rounds": 30,
                "huber_alpha": 0.9,
                "seed": 20260722,
                "class_weight": None,
            },
            "binary_objective": "binary",
            "binary_early_stopping_metric": "binary_logloss",
            "continuous_objective": "huber",
            "continuous_early_stopping_metric": "l2",
            "backend": "capability-tested LightGBM OpenCL/CUDA when selected; deterministic CPU reference otherwise",
            "gpu_opencl_double_precision": True,
            "gpu_acceleration_required_when_operator_explicitly_selects_an_accelerator": True,
            "model_reload_max_absolute_prediction_difference": 1e-12,
        },
        "calibration_contract": {
            "binary": "temperature-only scaling of clipped logits on tuning rows",
            "temperature_bounds": [0.25, 4.0],
            "temperature_optimizer": "bounded scalar minimization with xatol 1e-6",
            "continuous": "nonnegative slope through the origin on tuning rows",
            "continuous_slope_bounds": [0.0, 4.0],
            "retain_only_when_primary_tuning_loss_improves_by_more_than": 1e-12,
            "probability_clip": [1e-6, 0.999999],
            "classification_metric_threshold": 0.5,
            "threshold_has_trading_authority": False,
        },
        "evaluation_contract": {
            "primary_loss_metrics": {
                head: list(metrics) for head, metrics in PRIMARY_LOSS_METRICS.items()
            },
            "fdr_family_cardinality": 36,
            "fdr_family_order": "symbol BTC ETH SOL, then horizon 30 60 300, then head binary continuous, then each head's listed primary loss order",
            "permutation_draws": 10_000,
            "bootstrap_draws": 10_000,
            "permutation_unit": "UTC day",
            "permutation_statistic": "row-weighted mean paired loss difference with one random sign per UTC day",
            "bootstrap_unit": "UTC day",
            "bootstrap_estimand": "unweighted mean of finite per-day balanced accuracy, MCC, or Spearman values",
            "bootstrap_lower_quantile": 0.025,
            "fdr_family": "all spot_perpetual primary loss comparisons across three symbols, three horizons, two heads, and each declared primary loss metric",
            "fdr_method": "Benjamini-Hochberg",
            "loss_aggregation": "concatenate the six chronological out-of-sample test blocks and compute row-weighted losses",
            "fold_score_rule": "for binary log loss and Brier separately, spot_perpetual must beat that fold's training-prevalence baseline in at least four of six test folds",
            "prevalence_baseline": "positive-label fraction in that fold's primary training role, clipped to the probability bounds; the same value is reused for the five-second stress target",
            "continuous_controls": {
                "primary_gate": "zero return",
                "diagnostic": "that fold's mean primary training return",
            },
            "accuracy_rule": "pooled out-of-sample threshold-0.5 accuracy must strictly exceed pooled majority-class accuracy",
            "day_metric_rule": "compute each metric independently within each UTC day, discard only mathematically undefined day metrics, and take the unweighted mean across finite days",
            "bootstrap_rule": "sample the finite UTC-day metric values independently with replacement for 10000 draws and use the empirical 0.025 quantile of draw means",
            "stress_rule": "reuse each fitted and calibrated spot_perpetual primary model without refit; on stress-valid test rows, pooled log loss and Brier must each be strictly below the fold-specific primary-training-prevalence baseline",
            "seed_derivation": "LightGBM always uses 20260722; resampling uses 20260722 plus the zero-based canonical family index",
            "maximum_q_value": 0.05,
            "minimum_relative_improvement_spot_perpetual_vs_perpetual_only": 0.001,
            "minimum_binary_relative_improvement_vs_prevalence": 0.002,
            "minimum_continuous_mse_skill_vs_zero": 0.001,
            "minimum_positive_proper_score_folds": 4,
            "binary_day_block_lower_bounds": {"balanced_accuracy": 0.5, "MCC": 0.0},
            "continuous_day_block_lower_bound": {"Spearman": 0.0},
            "raw_accuracy_must_exceed_majority_prevalence": True,
            "stress_delay_binary_skill_vs_prevalence_must_be_positive": [
                "log_loss",
                "brier_score",
            ],
            "all_symbols_horizons_and_declared_primary_metrics_must_pass": True,
        },
        "resource_contract": {
            "feature_dtype": "float32",
            "target_and_metric_dtype": "float64",
            "raw_aggregate_trades_retained": False,
            "feature_cache_persisted_to_disk": False,
            "maximum_expected_in_memory_feature_bytes": 1_250_000_000,
            "LightGBM_histogram_pool_size_mb": 512,
        },
        "stopping_rule": {
            "primary_fail": "reject Round 72 without terminal holdout, PnL optimization, leverage, or larger model",
            "primary_pass": "open the three frozen terminal months with unchanged code and artifacts",
            "new_subsecond_hypothesis": "the June 2026 sub-second paper is materially new evidence and requires a separately numbered preregistration; it cannot rescue Round 72 post hoc",
        },
        "profitability_claim": False,
        "execution_or_fill_claim": False,
        "trading_authority": False,
        "leverage_authority": False,
    }
    return {
        **without_hash,
        "implementation_sha256": _canonical_sha256(without_hash),
    }


def load_round72_implementation(path: str | Path) -> dict[str, object]:
    """Load the exact reproducible implementation artifact and reject drift."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Round 72 implementation artifact is not an object")
    try:
        expected = build_round72_implementation_spec(
            design_sha256=str(value["design_sha256"]),
            inventory_sha256=str(value["inventory_sha256"]),
            inventory_file_sha256=str(value["inventory_file_sha256"]),
            frozen_at_utc=str(value["frozen_at_utc"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Round 72 implementation artifact fields are invalid") from exc
    if value != expected:
        raise ValueError("Round 72 implementation artifact differs from its generator")
    return value


def validate_layer_prefixes(
    feature_names: Sequence[str],
) -> tuple[int, int, int]:
    """Return immutable nested layer widths after exact-name validation."""

    names = tuple(str(value) for value in feature_names)
    expected = layer_feature_names("cross_asset")
    if names != expected:
        raise ValueError("Round 72 feature names differ from the implementation freeze")
    widths = tuple(len(layer_feature_names(layer)) for layer in FEATURE_LAYERS)
    if not widths[0] < widths[1] < widths[2]:
        raise RuntimeError("Round 72 feature layers are not strictly nested")
    return widths


__all__ = [
    "ANCHOR_SECOND_OFFSET",
    "ANCHOR_SPACING_SECONDS",
    "CLOCK_FEATURE_NAMES",
    "CROSS_ASSET_WINDOW_METRICS",
    "ENTRY_DELAYS_SECONDS",
    "FEATURE_BURN_IN_SECONDS",
    "FEATURE_LAYERS",
    "HORIZONS_SECONDS",
    "MARKET_CHANGE_METRICS",
    "MARKET_WINDOW_METRICS",
    "PAIR_WINDOW_METRICS",
    "PRIMARY_ENTRY_DELAY_SECONDS",
    "PRIMARY_LOSS_METRICS",
    "ROUND72_IMPLEMENTATION_SCHEMA",
    "ROUND72_IMPLEMENTATION_V1_SHA256",
    "SPOT_FLOW_LAGS_SECONDS",
    "STRESS_ENTRY_DELAY_SECONDS",
    "WINDOWS_SECONDS",
    "build_round72_implementation_spec",
    "cross_asset_feature_names",
    "layer_feature_names",
    "load_round72_implementation",
    "market_feature_names",
    "pair_feature_names",
    "validate_layer_prefixes",
]
