from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)
from simple_ai_trading.microstructure_model import (
    DeploymentRefitEvidence,
    MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
    MicrostructureModelArtifact,
    PerformanceConfidence,
    PrequentialValidationEvidence,
    PurgedSplitEvidence,
    ThresholdPolicy,
    ThresholdSearchEvidence,
    TerminalPrequentialEvidence,
    TradingMetrics,
    _apply_platt_scaling,
    _candidate_payload_sha256,
    _fit_platt_scaling,
    _model_strings_sha256,
    _performance_confidence,
    _purged_split,
    _select_threshold,
    _simulate_non_overlapping,
    _simulate_non_overlapping_trace,
    load_microstructure_action_scorer,
    load_microstructure_model_artifact,
    refit_validated_microstructure_model,
    save_microstructure_model_artifact,
)


def _five_day_dataset() -> MicrostructureDataset:
    timestamps = np.concatenate(
        [day * 86_400_000 + np.arange(1_000, dtype=np.int64) * 1_000 for day in range(5)]
    )
    rows = len(timestamps)
    exits = timestamps + 60_100
    ones = np.ones(rows, dtype=np.float64)
    return MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version="test",
        feature_names=("x", "y"),
        horizon_seconds=60,
        total_latency_ms=100,
        taker_fee_bps=5.0,
        reference_order_notional_quote=1.0,
        max_l1_participation=0.10,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=1,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=timestamps,
        long_exit_time_ms=exits,
        short_exit_time_ms=exits,
        features=np.zeros((rows, 2), dtype=np.float32),
        long_net_bps=-ones,
        short_net_bps=-ones,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(rows, dtype=np.int64),
        exit_quote_age_ms=np.zeros(rows, dtype=np.int64),
        entry_bid_price=100.0 * ones,
        entry_ask_price=101.0 * ones,
        fixed_exit_bid_price=100.0 * ones,
        fixed_exit_ask_price=101.0 * ones,
        entry_bid_qty=ones,
        entry_ask_qty=ones,
        fixed_exit_bid_qty=ones,
        fixed_exit_ask_qty=ones,
        long_l1_participation=0.01 * ones,
        short_l1_participation=0.01 * ones,
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
    )


def test_purged_split_separates_policy_selection_and_terminal_days() -> None:
    dataset = _five_day_dataset()
    splits, evidence = _purged_split(dataset)

    assert set(splits) == {"train", "tuning", "policy", "selection", "terminal"}
    assert all(len(indexes) >= 256 for indexes in splits.values())
    assert evidence.policy_rows == len(splits["policy"])
    assert evidence.selection_rows == len(splits["selection"])
    assert evidence.tuning_early_stop_rows >= 256
    assert evidence.tuning_calibration_rows >= 256
    assert evidence.tuning_internal_purged_rows > 0
    tuning_early_end = dataset.decision_time_ms[splits["tuning"]][
        evidence.tuning_early_stop_rows - 1
    ]
    assert tuning_early_end + evidence.purge_ms < evidence.tuning_calibration_start_ms
    for left, right in (("train", "tuning"), ("tuning", "policy"), ("policy", "selection")):
        assert (
            dataset.decision_time_ms[splits[left][-1]] + evidence.purge_ms
            < dataset.decision_time_ms[splits[right][0]]
        )
    assert (
        dataset.decision_time_ms[splits["selection"][-1]] + evidence.purge_ms
        < dataset.decision_time_ms[splits["terminal"][0]]
    )


def test_simulator_uses_eligible_side_and_realized_exit_time() -> None:
    metrics = _simulate_non_overlapping(
        timestamps=np.asarray([0, 1_000, 2_000], dtype=np.int64),
        long_exit_times=np.asarray([10_000, 5_000, 3_000], dtype=np.int64),
        short_exit_times=np.asarray([1_000, 5_000, 3_000], dtype=np.int64),
        long_targets=np.asarray([-9.0, 3.0, 100.0]),
        short_targets=np.asarray([2.0, -4.0, 100.0]),
        long_edge=np.asarray([10.0, 6.0, 8.0]),
        short_edge=np.asarray([5.0, 4.0, 7.0]),
        long_probability=np.asarray([0.1, 0.9, 0.9]),
        short_probability=np.asarray([0.9, 0.9, 0.9]),
        edge_threshold=1.0,
        probability_threshold=0.5,
    )

    assert metrics.trades == 2
    assert metrics.long_trades == 1
    assert metrics.short_trades == 1
    assert metrics.total_net_bps == 5.0


def test_simulator_never_executes_a_side_that_exceeds_the_l1_participation_gate() -> None:
    metrics = _simulate_non_overlapping(
        timestamps=np.asarray([0], dtype=np.int64),
        long_exit_times=np.asarray([1_000], dtype=np.int64),
        short_exit_times=np.asarray([1_000], dtype=np.int64),
        long_targets=np.asarray([50.0]),
        short_targets=np.asarray([2.0]),
        long_edge=np.asarray([100.0]),
        short_edge=np.asarray([3.0]),
        long_probability=np.asarray([0.99]),
        short_probability=np.asarray([0.90]),
        edge_threshold=1.0,
        probability_threshold=0.5,
        long_eligible=np.asarray([False]),
        short_eligible=np.asarray([True]),
    )

    assert metrics.trades == 1
    assert metrics.long_trades == 0
    assert metrics.short_trades == 1
    assert metrics.total_net_bps == 2.0


def test_day_block_confidence_includes_the_full_calendar_and_is_deterministic() -> None:
    timestamps = np.arange(60, dtype=np.int64) * 86_400_000
    ones = np.ones(60, dtype=np.float64)
    zeros = np.zeros(60, dtype=np.float64)
    trace = _simulate_non_overlapping_trace(
        timestamps=timestamps,
        long_exit_times=timestamps + 1_000,
        short_exit_times=timestamps + 1_000,
        long_targets=ones,
        short_targets=-ones,
        long_edge=ones,
        short_edge=zeros,
        long_probability=ones,
        short_probability=zeros,
        edge_threshold=0.5,
        probability_threshold=0.5,
    )

    first = _performance_confidence(trace, timestamps)
    second = _performance_confidence(trace, timestamps)

    assert first == second
    assert first.calendar_days == 60
    assert first.active_days == 60
    assert first.bootstrap_samples == 2_000
    assert first.mean_daily_net_bps_ci_lower == pytest.approx(1.0)
    assert first.bootstrap_probability_mean_positive == 1.0


def test_platt_scaling_corrects_probability_base_rate() -> None:
    labels = np.concatenate((np.zeros(800), np.ones(200)))
    raw = np.full(1_000, 0.40)

    calibration = _fit_platt_scaling(raw, labels)
    calibrated = _apply_platt_scaling(raw, calibration)

    assert calibration[0] > 0.0
    assert np.isclose(np.mean(calibrated), 0.20, atol=1e-4)
    assert np.mean(np.square(calibrated - labels)) < np.mean(np.square(raw - labels))


def test_threshold_search_uses_base_rate_relative_probability_tails() -> None:
    rows = 500
    timestamps = np.arange(rows, dtype=np.int64) * 1_000
    long_edge = np.full(rows, -1.0)
    long_edge[-100:] = np.linspace(0.1, 1.0, 100)
    long_probability = np.full(rows, 0.20)
    long_probability[-100:] = np.linspace(0.25, 0.35, 100)
    long_targets = np.full(rows, -2.0)
    long_targets[-100:] = 3.0

    policy, metrics, evidence = _select_threshold(
        risk_level="conservative",
        timestamps=timestamps,
        long_exit_times=timestamps + 1_000,
        short_exit_times=timestamps + 1_000,
        long_targets=long_targets,
        short_targets=np.full(rows, -2.0),
        long_edge=long_edge,
        short_edge=np.full(rows, -1.0),
        long_probability=long_probability,
        short_probability=np.full(rows, 0.20),
    )

    assert policy.minimum_profitable_probability < 0.50
    assert metrics.trades >= 20
    assert metrics.total_net_bps > 0.0
    assert evidence.minimum_required_trades == 20
    assert evidence.policies_meeting_trade_minimum > 0


def _runtime_artifact_payload(*, status: str = "accepted") -> dict[str, object]:
    metrics = {
        "trades": 40,
        "total_net_bps": 100.0,
        "mean_net_bps": 2.5,
        "median_net_bps": 2.0,
        "win_rate": 0.6,
        "profit_factor": 1.5,
        "max_drawdown_bps": 20.0,
        "worst_trade_bps": -5.0,
        "best_trade_bps": 8.0,
        "long_trades": 20,
        "short_trades": 20,
        "active_days": 20,
        "trades_per_active_day": 2.0,
    }
    no_trade = {
        "trades": 0,
        "total_net_bps": 0.0,
        "mean_net_bps": 0.0,
        "median_net_bps": 0.0,
        "win_rate": 0.0,
        "profit_factor": None,
        "max_drawdown_bps": 0.0,
        "worst_trade_bps": 0.0,
        "best_trade_bps": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "active_days": 0,
        "trades_per_active_day": 0.0,
    }
    payload: dict[str, object] = {
        "schema_version": MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
        "model_family": "side_specific_hurdle_expected_value",
        "status": status,
        "rejection_reasons": [] if status == "accepted" else ["selection_failed"],
        "symbol": "BTCUSDT",
        "risk_level": "conservative",
        "feature_version": MICROSTRUCTURE_FEATURE_VERSION,
        "feature_names": list(MICROSTRUCTURE_FEATURE_NAMES),
        "horizon_seconds": 60,
        "total_latency_ms": 250,
        "taker_fee_bps": 5.0,
        "reference_order_notional_quote": 1_000.0,
        "max_l1_participation": 0.10,
        "max_quote_age_ms": 1_000,
        "decision_cadence_seconds": 5,
        "target_mode": "exchange_trigger_market_exit_1s_adverse_first",
        "stop_loss_bps": 25.0,
        "take_profit_bps": 40.0,
        "trigger_execution_slippage_bps": 1.0,
        "path_resolution_ms": 1_000,
        "training_backend_kind": "opencl",
        "training_backend_device": "opencl:0:0",
        "lightgbm_version": "4.6.0",
        "seed": 17,
        "unique_utc_days": 365,
        "calendar_span_days": 365,
        "minimum_promotion_days": 365,
        "deployment_calibration_days": 14,
        "maximum_model_age_seconds": 86_400,
        "calendar_day_coverage_ratio": 1.0,
        "minimum_rows_per_utc_day": 100,
        "daily_rows_p10": 100.0,
        "median_rows_per_utc_day": 100.0,
        "split": {
            "train_rows": 55_000,
            "tuning_rows": 15_000,
            "policy_rows": 10_000,
            "selection_rows": 10_000,
            "terminal_rows": 10_000,
            "train_end_ms": 1_715_000_000_000,
            "tuning_start_ms": 1_715_000_001_000,
            "policy_start_ms": 1_720_000_000_000,
            "selection_start_ms": 1_725_000_000_000,
            "terminal_start_ms": 1_730_000_000_000,
            "purge_ms": 61_000,
            "purged_rows": 100,
            "tuning_early_stop_rows": 7_000,
            "tuning_calibration_rows": 7_000,
            "tuning_calibration_start_ms": 1_717_500_000_000,
            "tuning_internal_purged_rows": 1_000,
        },
        "terminal_evaluated_at": "2026-01-01T00:00:00+00:00",
        "terminal_metrics": dict(metrics),
        "selection_confidence": {
            "calendar_days": 55,
            "active_days": 40,
            "block_length_days": 4,
            "bootstrap_samples": 2_000,
            "mean_daily_net_bps": 2.0,
            "median_daily_net_bps": 1.0,
            "worst_daily_net_bps": -3.0,
            "best_daily_net_bps": 8.0,
            "annualized_daily_sharpe": 2.2,
            "mean_daily_net_bps_ci_lower": 0.4,
            "mean_daily_net_bps_ci_upper": 3.5,
            "bootstrap_probability_mean_positive": 0.99,
        },
        "terminal_confidence": {
            "calendar_days": 55,
            "active_days": 40,
            "block_length_days": 4,
            "bootstrap_samples": 2_000,
            "mean_daily_net_bps": 1.8,
            "median_daily_net_bps": 0.9,
            "worst_daily_net_bps": -4.0,
            "best_daily_net_bps": 7.0,
            "annualized_daily_sharpe": 1.9,
            "mean_daily_net_bps_ci_lower": 0.2,
            "mean_daily_net_bps_ci_upper": 3.2,
            "bootstrap_probability_mean_positive": 0.98,
        },
        "terminal_auc": {"long": 0.60, "short": 0.58},
        "terminal_brier": {"long": 0.20, "short": 0.21},
        "tuning_auc": {"long": 0.61, "short": 0.59},
        "tuning_brier": {"long": 0.19, "short": 0.20},
        "selection_auc": {"long": 0.60, "short": 0.58},
        "selection_brier": {"long": 0.20, "short": 0.21},
        "policy_metrics": dict(metrics),
        "selection_metrics": dict(metrics),
        "selection_baselines": {"no_trade": dict(no_trade)},
        "terminal_baselines": {"no_trade": dict(no_trade)},
        "dataset_summary": {
            "trade_feature_embargo_ms": 1_000,
            "reference_order_notional_quote": 1_000.0,
            "max_l1_participation": 0.10,
            "max_quote_age_ms": 1_000,
            "decision_cadence_seconds": 5,
            "source_evidence": {
                "build_id": "c" * 64,
                "schema_version": "binance-usdm-tick-v4",
                "feature_build_version": "book-ticker-event-time-v1",
                "availability_clock": "event_time_ms",
                "symbol": "BTCUSDT",
                "status": "complete",
                "is_current": True,
                "manifest_fingerprint": "d" * 64,
                "source_archive_count": 365,
                "source_manifest_rows": 1_000_000,
                "source_raw_rows": 1_000_000,
                "feature_rows": 100_000,
                "first_feature_second_ms": 1_700_000_000_000,
                "last_feature_second_ms": 1_731_535_999_000,
                "duplicate_seconds": 0,
                "invalid_feature_rows": 0,
                "quote_update_sum": 1_000_000,
                "verified": True,
                "manifest_current": True,
            }
        },
        "threshold_policy": {
            "minimum_predicted_edge_bps": 1.0,
            "minimum_profitable_probability": 0.60,
            "selection_utility_bps": 1.0,
        },
        "policy_search": {
            "rows": 10_000,
            "long_positive_edge_rows": 2_000,
            "short_positive_edge_rows": 2_000,
            "minimum_required_trades": 20,
            "evaluated_policy_count": 25,
            "policies_meeting_trade_minimum": 10,
            "best_observed_utility_bps": 60.0,
            "best_observed_metrics": dict(metrics),
        },
        "probability_calibration": {"long": [1.0, 0.0], "short": [1.0, 0.0]},
        "trained_at": "2025-12-01T00:00:00+00:00",
    }
    models = {
        "long_probability": "0.80",
        "long_win_magnitude": "4.0",
        "long_loss_magnitude": "1.0",
        "short_probability": "0.70",
        "short_win_magnitude": "3.0",
        "short_loss_magnitude": "2.0",
    }
    payload["model_strings"] = models
    payload["best_iterations"] = {name: 1 for name in models}
    payload["deployment_model_strings"] = dict(models)
    training_cutoff_ms = 1_731_535_999_000
    payload["deployment_refit"] = {
        "refit_mode": "full_history_fixed_hyperparameters",
        "backend_kind": "opencl",
        "backend_device": "opencl:0:0",
        "lightgbm_version": "4.6.0",
        "training_rows": 100_000,
        "calibration_days": 14,
        "calibration_start_ms": training_cutoff_ms - 14 * 86_400_000,
        "calibration_end_ms": training_cutoff_ms,
        "side_training_rows": {"long": 99_000, "short": 99_000},
        "side_calibration_rows": {"long": 13_000, "short": 13_000},
        "probability_calibration": {
            "long": [1.0, 0.0],
            "short": [1.0, 0.0],
        },
        "training_cutoff_ms": training_cutoff_ms,
        "maximum_model_age_seconds": 86_400,
        "expires_at_ms": training_cutoff_ms + 86_400_000,
        "source_feature_build_id": "c" * 64,
        "source_manifest_fingerprint": "d" * 64,
        "validation_model_sha256": _model_strings_sha256(models),
        "deployment_model_sha256": _model_strings_sha256(models),
        "fitted_at": "2024-11-13T22:13:19+00:00",
    }
    candidate_payload = json.loads(json.dumps(payload))
    candidate_payload.update(
        {
            "status": "candidate",
            "rejection_reasons": [],
            "terminal_auc": None,
            "terminal_brier": None,
            "terminal_metrics": None,
            "terminal_confidence": None,
            "terminal_baselines": None,
            "deployment_model_strings": None,
            "deployment_refit": None,
            "terminal_evaluated_at": None,
            "prequential_validation": None,
            "terminal_prequential": None,
            "shadow_validation": None,
        }
    )
    candidate_sha = hashlib.sha256(
        json.dumps(
            candidate_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    payload["prequential_validation"] = {
        "version": "microstructure-prequential-fixed-refit-v1",
        "report_sha256": "e" * 64,
        "predictions_sha256": "f" * 64,
        "chart_sha256": "1" * 64,
        "candidate_sha256": candidate_sha,
        "protocol_sha256": "2" * 64,
        "fold_models_sha256": "3" * 64,
        "source_feature_build_id": "c" * 64,
        "source_manifest_fingerprint": "d" * 64,
        "generated_at_ms": 1_765_000_000_000,
        "planned_folds": 8,
        "complete_folds": 8,
        "evaluated_rows": 10_000,
        "selection_coverage_ratio": 1.0,
        "total_net_bps": 100.0,
        "profit_factor": 1.5,
        "max_drawdown_bps": 20.0,
        "mean_daily_net_bps_ci_lower": 0.4,
        "attached_at": "2026-01-01T00:00:00+00:00",
    }
    terminal_start_ms = 1_730_000_000_000
    terminal_folds = [
        {
            "fold": fold,
            "status": "complete",
            "error": None,
            "backend_kind": "opencl",
            "backend_device": "opencl:0:0",
            "evaluation_start_ms": terminal_start_ms + (fold - 1) * 1_000,
            "evaluation_end_exclusive_ms": terminal_start_ms + fold * 1_000,
            "evaluation_rows": 3_334 if fold == 3 else 3_333,
            "evaluation_metrics": {"total_net_bps": 10.0},
            "policy": dict(payload["threshold_policy"]),
            "model_sha256": str(fold + 3) * 64,
        }
        for fold in range(1, 4)
    ]
    terminal_fold_models_sha = hashlib.sha256(
        json.dumps(
            [
                {"fold": fold["fold"], "model_sha256": fold["model_sha256"]}
                for fold in terminal_folds
            ],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    payload["terminal_prequential"] = {
        "version": "microstructure-prequential-fixed-refit-v1",
        "protocol_sha256": "2" * 64,
        "fold_models_sha256": terminal_fold_models_sha,
        "backend_kind": "opencl",
        "backend_device": "opencl:0:0",
        "planned_folds": 3,
        "complete_folds": 3,
        "expected_rows": 10_000,
        "evaluated_rows": 10_000,
        "first_evaluation_ms": terminal_start_ms,
        "last_evaluation_ms": terminal_start_ms + 2_999,
        "latest_policy": dict(payload["threshold_policy"]),
        "folds": terminal_folds,
    }
    shadow_start_ms = training_cutoff_ms + 1_000
    shadow_complete_ms = shadow_start_ms + 21_600_000
    payload["shadow_validation"] = {
        "version": "microstructure-public-feed-shadow-v1",
        "report_sha256": "7" * 64,
        "trades_sha256": "8" * 64,
        "capture_manifest_sha256": "9" * 64,
        "raw_capture_sha256": "a" * 64,
        "candidate_sha256": candidate_sha,
        "deployment_model_sha256": _model_strings_sha256(models),
        "symbol": "BTCUSDT",
        "provider": "binance_public_usdm_websocket",
        "clock_offset_ms": 0.0,
        "started_at_ms": shadow_start_ms,
        "completed_at_ms": shadow_complete_ms,
        "duration_seconds": 21_600.0,
        "decisions": 200,
        "actionable_decisions": 100,
        "virtual_trades": 40,
        "long_trades": 20,
        "short_trades": 20,
        "execution_liquidity_rejections": 0,
        "expired_entries": 0,
        "pending_entries_at_end": 0,
        "end_censored_signals": 1,
        "total_net_bps": 100.0,
        "profit_factor": 1.5,
        "max_drawdown_bps": 20.0,
        "feed_sequence_gaps": 0,
        "invalid_events": 0,
        "late_event_resets": 0,
        "feature_gap_resets": 0,
        "deadline_misses": 0,
        "inference_failures": 0,
        "forced_closes": 0,
        "orders_submitted": 0,
        "attached_at": "2024-11-14T04:13:21+00:00",
    }
    return payload


def test_promoted_runtime_scorer_loads_exact_contract_and_selects_best_side(
    tmp_path, monkeypatch
) -> None:
    class _Booster:
        def __init__(self, *, model_str: str) -> None:
            self.value = float(model_str)

        def predict(self, rows):
            return np.full(len(rows), self.value, dtype=np.float64)

    monkeypatch.setattr("simple_ai_trading.microstructure_model.lgb.Booster", _Booster)
    path = tmp_path / "microstructure-model.json"
    path.write_text(json.dumps(_runtime_artifact_payload()), encoding="utf-8")

    scorer = load_microstructure_action_scorer(path)
    features = np.zeros(len(MICROSTRUCTURE_FEATURE_NAMES), dtype=np.float32)
    result = scorer.score(
        features,
        decision_time_ms=10_000,
        order_notional_quote=1_000.0,
        close_bid=50_000.0,
        close_ask=50_001.0,
        close_bid_qty=1.0,
        close_ask_qty=1.0,
        quote_time_ms=9_900,
        observation_time_ms=10_100,
    )
    oversized = scorer.score(
        features,
        decision_time_ms=10_000,
        order_notional_quote=1_001.0,
        close_bid=50_000.0,
        close_ask=50_001.0,
        close_bid_qty=1.0,
        close_ask_qty=1.0,
        quote_time_ms=9_900,
        observation_time_ms=10_100,
    )

    assert scorer.symbol == "BTCUSDT"
    assert len(scorer.artifact_sha256) == 64
    assert result.side == "LONG"
    assert result.long_expected_net_bps == pytest.approx(3.0)
    assert result.short_expected_net_bps == pytest.approx(1.5)
    assert result.long_profitable_probability == pytest.approx(0.8)
    assert result.long_l1_participation < 0.10
    assert oversized.side == "FLAT"
    assert oversized.reason == "order_notional_exceeds_validated_reference"
    stale = scorer.score(
        features,
        decision_time_ms=10_000,
        order_notional_quote=1_000.0,
        close_bid=50_000.0,
        close_ask=50_001.0,
        close_bid_qty=1.0,
        close_ask_qty=1.0,
        quote_time_ms=8_000,
        observation_time_ms=10_100,
    )
    assert stale.side == "FLAT"
    assert stale.reason == "quote_age_exceeds_validated_limit"
    assert scorer.enforce_model_freshness is True
    with pytest.raises(ValueError, match="violates decision cadence"):
        scorer.score(
            features,
            decision_time_ms=11_000,
            order_notional_quote=1_000.0,
            close_bid=50_000.0,
            close_ask=50_001.0,
            close_bid_qty=1.0,
            close_ask_qty=1.0,
            quote_time_ms=10_900,
            observation_time_ms=11_100,
        )
    with pytest.raises(RuntimeError, match="expired"):
        scorer.score(
            features,
            decision_time_ms=1_731_622_400_000,
            order_notional_quote=1_000.0,
            close_bid=50_000.0,
            close_ask=50_001.0,
            close_bid_qty=1.0,
            close_ask_qty=1.0,
            quote_time_ms=1_731_622_399_900,
            observation_time_ms=1_731_622_400_100,
        )
    with pytest.raises(RuntimeError, match="expired"):
        load_microstructure_action_scorer(
            path,
            as_of_ms=1_731_622_400_000,
        )


def test_runtime_scorer_rejects_unpromoted_or_drifted_artifacts(tmp_path) -> None:
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps(_runtime_artifact_payload(status="rejected")), encoding="utf-8")
    with pytest.raises(ValueError, match="rejected"):
        load_microstructure_action_scorer(rejected)

    drifted_payload = _runtime_artifact_payload()
    drifted_payload["feature_names"] = [*MICROSTRUCTURE_FEATURE_NAMES[:-1], "drifted"]
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(drifted_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature names"):
        load_microstructure_action_scorer(drifted)

    unproven_payload = _runtime_artifact_payload()
    unproven_payload["dataset_summary"]["source_evidence"] = None  # type: ignore[index]
    unproven = tmp_path / "unproven.json"
    unproven.write_text(json.dumps(unproven_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source provenance"):
        load_microstructure_action_scorer(unproven)

    no_prequential_payload = _runtime_artifact_payload()
    no_prequential_payload["prequential_validation"] = None
    no_prequential = tmp_path / "no-prequential.json"
    no_prequential.write_text(json.dumps(no_prequential_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prequential validation evidence"):
        load_microstructure_action_scorer(no_prequential)

    no_shadow_payload = _runtime_artifact_payload()
    no_shadow_payload["shadow_validation"] = None
    no_shadow = tmp_path / "no-shadow.json"
    no_shadow.write_text(json.dumps(no_shadow_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no-order shadow evidence"):
        load_microstructure_action_scorer(no_shadow)

    unsafe_shadow_payload = _runtime_artifact_payload()
    unsafe_shadow_payload["shadow_validation"]["pending_entries_at_end"] = 1  # type: ignore[index]
    unsafe_shadow = tmp_path / "unsafe-shadow.json"
    unsafe_shadow.write_text(json.dumps(unsafe_shadow_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="shadow promotion gates"):
        load_microstructure_action_scorer(unsafe_shadow)

    unprotected_payload = _runtime_artifact_payload()
    unprotected_payload["target_mode"] = "fixed_horizon"
    unprotected = tmp_path / "unprotected.json"
    unprotected.write_text(json.dumps(unprotected_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protective-exit contract"):
        load_microstructure_action_scorer(unprotected)


def test_terminal_validated_artifact_requires_a_source_bound_expiring_deployment_refit(
    monkeypatch,
    tmp_path,
) -> None:
    days = 30
    rows_per_day = 300
    timestamps = np.concatenate(
        [
            1_700_000_000_000
            + day * 86_400_000
            + np.arange(rows_per_day, dtype=np.int64) * 1_000
            for day in range(days)
        ]
    )
    rows = len(timestamps)
    values = np.ones(rows, dtype=np.float64)
    alternating = np.where(np.arange(rows) % 2 == 0, 2.0, -2.0)
    source = {
        "build_id": "a" * 64,
        "schema_version": "binance-usdm-tick-v4",
        "feature_build_version": "book-ticker-event-time-v1",
        "availability_clock": "event_time_ms",
        "symbol": "BTCUSDT",
        "status": "complete",
        "is_current": True,
        "manifest_fingerprint": "b" * 64,
        "source_archive_count": days,
        "source_manifest_rows": rows * 10,
        "source_raw_rows": rows * 10,
        "feature_rows": rows,
        "first_feature_second_ms": int(timestamps[0]),
        "last_feature_second_ms": int(timestamps[-1]),
        "duplicate_seconds": 0,
        "invalid_feature_rows": 0,
        "quote_update_sum": rows * 10,
        "verified": True,
        "manifest_current": True,
    }
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=60,
        total_latency_ms=250,
        taker_fee_bps=5.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=0.05,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=1,
        target_mode="exchange_trigger_market_exit_1s_adverse_first",
        stop_loss_bps=25.0,
        take_profit_bps=40.0,
        trigger_execution_slippage_bps=1.0,
        path_resolution_ms=1_000,
        decision_time_ms=timestamps,
        long_exit_time_ms=timestamps + 60_000,
        short_exit_time_ms=timestamps + 60_000,
        features=np.zeros((rows, len(MICROSTRUCTURE_FEATURE_NAMES)), dtype=np.float32),
        long_net_bps=alternating,
        short_net_bps=-alternating,
        entry_spread_bps=values,
        exit_spread_bps=values,
        entry_quote_age_ms=np.zeros(rows, dtype=np.int64),
        exit_quote_age_ms=np.zeros(rows, dtype=np.int64),
        entry_bid_price=50_000.0 * values,
        entry_ask_price=50_001.0 * values,
        fixed_exit_bid_price=50_000.0 * values,
        fixed_exit_ask_price=50_001.0 * values,
        entry_bid_qty=values,
        entry_ask_qty=values,
        fixed_exit_bid_qty=values,
        fixed_exit_ask_qty=values,
        long_l1_participation=0.01 * values,
        short_l1_participation=0.01 * values,
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
        source_evidence=source,
    )
    metrics = TradingMetrics(40, 100.0, 2.5, 2.0, 0.6, 1.5, 20.0, -5.0, 8.0, 20, 20, 20, 2.0)
    confidence = PerformanceConfidence(20, 20, 3, 2_000, 5.0, 4.0, -2.0, 12.0, 2.0, 0.5, 8.0, 0.99)
    model_strings = {
        name: f"validation-{name}"
        for name in (
            "long_probability",
            "long_win_magnitude",
            "long_loss_magnitude",
            "short_probability",
            "short_win_magnitude",
            "short_loss_magnitude",
        )
    }
    artifact = MicrostructureModelArtifact(
        schema_version=MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
        model_family="side_specific_hurdle_expected_value",
        status="validated",
        rejection_reasons=(),
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        risk_level="conservative",
        horizon_seconds=60,
        total_latency_ms=250,
        taker_fee_bps=5.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=0.05,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=1,
        target_mode="exchange_trigger_market_exit_1s_adverse_first",
        stop_loss_bps=25.0,
        take_profit_bps=40.0,
        trigger_execution_slippage_bps=1.0,
        path_resolution_ms=1_000,
        training_backend_kind="cpu",
        training_backend_device="cpu",
        lightgbm_version="test",
        seed=7,
        unique_utc_days=days,
        calendar_span_days=days,
        calendar_day_coverage_ratio=1.0,
        minimum_rows_per_utc_day=rows_per_day,
        daily_rows_p10=float(rows_per_day),
        median_rows_per_utc_day=float(rows_per_day),
        minimum_promotion_days=1,
        deployment_calibration_days=2,
        maximum_model_age_seconds=3_600,
        split=PurgedSplitEvidence(
            2_000,
            1_000,
            1_000,
            1_000,
            1_000,
            1,
            2,
            3,
            4,
            5,
            60_000,
            3_000,
            450,
            500,
            4,
            50,
        ),
        best_iterations={name: 1 for name in model_strings},
        probability_calibration={"long": (1.0, 0.0), "short": (1.0, 0.0)},
        threshold_policy=ThresholdPolicy(1.0, 0.6, 10.0),
        policy_search=ThresholdSearchEvidence(1_000, 100, 100, 20, 10, 5, 10.0, metrics),
        tuning_auc={"long": 0.6, "short": 0.6},
        tuning_brier={"long": 0.2, "short": 0.2},
        selection_auc={"long": 0.6, "short": 0.6},
        selection_brier={"long": 0.2, "short": 0.2},
        terminal_auc={"long": 0.6, "short": 0.6},
        terminal_brier={"long": 0.2, "short": 0.2},
        policy_metrics=metrics,
        selection_metrics=metrics,
        terminal_metrics=metrics,
        selection_confidence=confidence,
        terminal_confidence=confidence,
        selection_baselines={"no_trade": TradingMetrics(0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)},
        terminal_baselines={"no_trade": TradingMetrics(0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)},
        model_strings=model_strings,
        deployment_model_strings=None,
        deployment_refit=None,
        dataset_summary=dataset.summary(),
        trained_at=datetime.now(timezone.utc).isoformat(),
        terminal_evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
    terminal_folds = tuple(
        {
            "fold": fold,
            "status": "complete",
            "error": None,
            "backend_kind": "cpu",
            "backend_device": "cpu",
            "evaluation_start_ms": 4 + fold,
            "evaluation_end_exclusive_ms": 5 + fold,
            "evaluation_rows": 334 if fold == 3 else 333,
            "evaluation_metrics": {"total_net_bps": 10.0},
            "policy": {
                "minimum_predicted_edge_bps": 1.0,
                "minimum_profitable_probability": 0.6,
                "selection_utility_bps": 10.0,
            },
            "model_sha256": str(fold + 3) * 64,
        }
        for fold in range(1, 4)
    )
    terminal_fold_sha = hashlib.sha256(
        json.dumps(
            [
                {"fold": fold["fold"], "model_sha256": fold["model_sha256"]}
                for fold in terminal_folds
            ],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    artifact = replace(
        artifact,
        prequential_validation=PrequentialValidationEvidence(
            version="microstructure-prequential-fixed-refit-v1",
            report_sha256="d" * 64,
            predictions_sha256="e" * 64,
            chart_sha256="f" * 64,
            candidate_sha256=_candidate_payload_sha256(artifact),
            protocol_sha256="1" * 64,
            fold_models_sha256="2" * 64,
            source_feature_build_id="a" * 64,
            source_manifest_fingerprint="b" * 64,
            generated_at_ms=1_700_000_000_000,
            planned_folds=5,
            complete_folds=5,
            evaluated_rows=1_000,
            selection_coverage_ratio=1.0,
            total_net_bps=100.0,
            profit_factor=1.5,
            max_drawdown_bps=20.0,
            mean_daily_net_bps_ci_lower=0.5,
            attached_at=datetime.now(timezone.utc).isoformat(),
        ),
        terminal_prequential=TerminalPrequentialEvidence(
            version="microstructure-prequential-fixed-refit-v1",
            protocol_sha256="1" * 64,
            fold_models_sha256=terminal_fold_sha,
            backend_kind="cpu",
            backend_device="cpu",
            planned_folds=3,
            complete_folds=3,
            expected_rows=1_000,
            evaluated_rows=1_000,
            first_evaluation_ms=5,
            last_evaluation_ms=7,
            latest_policy=ThresholdPolicy(1.0, 0.6, 10.0),
            folds=terminal_folds,
        ),
    )

    class _Booster:
        def __init__(self, identity: int) -> None:
            self.identity = identity

        def predict(self, features):
            return np.full(len(features), 0.5, dtype=np.float64)

        def model_to_string(self, *, num_iteration: int) -> str:
            return f"deployment-{self.identity}-iteration-{num_iteration}"

    calls: list[int] = []

    def fake_train_fixed_booster(**_kwargs):
        calls.append(len(calls))
        return _Booster(calls[-1])

    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model._train_fixed_booster",
        fake_train_fixed_booster,
    )
    monkeypatch.setattr(
        "simple_ai_trading.microstructure_model._backend_parameters",
        lambda _backend, _seed: ({"device_type": "cpu"}, "cpu", "cpu"),
    )

    shadow_candidate = refit_validated_microstructure_model(
        artifact,
        dataset,
        compute_backend="cpu",
    )

    assert shadow_candidate.status == "shadow_candidate"
    assert shadow_candidate.promotion_eligible is False
    assert shadow_candidate.shadow_validation is None
    assert len(calls) == 8
    assert shadow_candidate.deployment_model_strings is not None
    assert set(shadow_candidate.deployment_model_strings) == set(model_strings)
    assert isinstance(shadow_candidate.deployment_refit, DeploymentRefitEvidence)
    assert shadow_candidate.deployment_refit.training_rows == rows
    assert shadow_candidate.deployment_refit.calibration_days == 2
    assert shadow_candidate.deployment_refit.expires_at_ms == timestamps[-1] + 3_600_000
    assert shadow_candidate.deployment_refit.validation_model_sha256 == _model_strings_sha256(
        model_strings
    )
    path = tmp_path / "shadow-candidate-microstructure.json"
    save_microstructure_model_artifact(shadow_candidate, path)
    reloaded = load_microstructure_model_artifact(path)
    assert reloaded.status == "shadow_candidate"
    assert reloaded.max_quote_age_ms == 1_000
    assert reloaded.deployment_refit is not None
    assert (
        reloaded.deployment_refit.deployment_model_sha256
        == shadow_candidate.deployment_refit.deployment_model_sha256
    )
