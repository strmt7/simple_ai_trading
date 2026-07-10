from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from simple_ai_trading.microstructure_features import MicrostructureDataset
from simple_ai_trading.microstructure_model import (
    MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
    MicrostructureModelArtifact,
    PerformanceConfidence,
    PurgedSplitEvidence,
    ThresholdPolicy,
    ThresholdSearchEvidence,
    TradingMetrics,
    evaluate_microstructure_model_terminal,
)
from simple_ai_trading import microstructure_prequential as prequential
from simple_ai_trading.microstructure_prequential import (
    PrequentialConfig,
    attach_verified_prequential_evidence,
    evaluate_prequential_microstructure_model,
    plan_prequential_folds,
)
from simple_ai_trading.microstructure_warehouse import (
    BOOK_TICKER_FEATURE_BUILD_VERSION,
    TICK_WAREHOUSE_SCHEMA_VERSION,
)


_DAY_MS = 86_400_000


def _dataset_and_artifact() -> tuple[MicrostructureDataset, MicrostructureModelArtifact]:
    days = 260
    rows_per_day = 80
    first_day = 20_000
    timestamps = np.concatenate(
        [
            (first_day + day) * _DAY_MS
            + np.arange(rows_per_day, dtype=np.int64) * 1_000
            for day in range(days)
        ]
    )
    rows = len(timestamps)
    alternating = np.where(np.arange(rows) % 2 == 0, 2.0, -2.0)
    ones = np.ones(rows, dtype=np.float64)
    source = {
        "build_id": "a" * 64,
        "schema_version": TICK_WAREHOUSE_SCHEMA_VERSION,
        "feature_build_version": BOOK_TICKER_FEATURE_BUILD_VERSION,
        "availability_clock": "event_time_ms",
        "symbol": "BTCUSDT",
        "status": "complete",
        "is_current": True,
        "manifest_fingerprint": "b" * 64,
        "source_archive_count": days,
        "source_manifest_rows": rows * 2,
        "source_raw_rows": rows * 2,
        "feature_rows": rows,
        "first_feature_second_ms": int(timestamps[0]),
        "last_feature_second_ms": int(timestamps[-1]),
        "duplicate_seconds": 0,
        "invalid_feature_rows": 0,
        "quote_update_sum": rows * 2,
        "verified": True,
        "manifest_current": True,
    }
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version="fixture-v1",
        feature_names=("x", "y"),
        horizon_seconds=1,
        total_latency_ms=0,
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
        long_exit_time_ms=timestamps + 1_000,
        short_exit_time_ms=timestamps + 1_000,
        features=np.column_stack(
            (
                (alternating > 0.0).astype(np.float32),
                (alternating < 0.0).astype(np.float32),
            )
        ),
        long_net_bps=alternating,
        short_net_bps=-alternating,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(rows, dtype=np.int64),
        exit_quote_age_ms=np.zeros(rows, dtype=np.int64),
        entry_bid_price=50_000.0 * ones,
        entry_ask_price=50_001.0 * ones,
        fixed_exit_bid_price=50_000.0 * ones,
        fixed_exit_ask_price=50_001.0 * ones,
        entry_bid_qty=ones,
        entry_ask_qty=ones,
        fixed_exit_bid_qty=ones,
        fixed_exit_ask_qty=ones,
        long_l1_participation=0.01 * ones,
        short_l1_participation=0.01 * ones,
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
        source_evidence=source,
    )
    empty_metrics = TradingMetrics(
        0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0, 0, 0, 0.0
    )
    confidence = PerformanceConfidence(
        10, 0, 3, 2_000, 0.0, 0.0, 0.0, 0.0, None, -1.0, 1.0, 0.5
    )
    names = {
        f"{side}_{component}": 1
        for side in ("long", "short")
        for component in ("probability", "win_magnitude", "loss_magnitude")
    }
    selection_start_ms = (first_day + 220) * _DAY_MS
    terminal_start_ms = (first_day + 250) * _DAY_MS
    artifact = MicrostructureModelArtifact(
        schema_version=MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
        model_family="side_specific_hurdle_expected_value",
        status="candidate",
        rejection_reasons=(),
        symbol=dataset.symbol,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        risk_level="conservative",
        horizon_seconds=dataset.horizon_seconds,
        total_latency_ms=dataset.total_latency_ms,
        taker_fee_bps=dataset.taker_fee_bps,
        reference_order_notional_quote=dataset.reference_order_notional_quote,
        max_l1_participation=dataset.max_l1_participation,
        max_quote_age_ms=dataset.max_quote_age_ms,
        decision_cadence_seconds=dataset.decision_cadence_seconds,
        target_mode=dataset.target_mode,
        stop_loss_bps=dataset.stop_loss_bps,
        take_profit_bps=dataset.take_profit_bps,
        trigger_execution_slippage_bps=dataset.trigger_execution_slippage_bps,
        path_resolution_ms=dataset.path_resolution_ms,
        training_backend_kind="cpu",
        training_backend_device="cpu",
        lightgbm_version=str(lgb.__version__),
        seed=17,
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
            train_rows=10_000,
            tuning_rows=2_000,
            policy_rows=2_000,
            selection_rows=2_400,
            terminal_rows=800,
            train_end_ms=(first_day + 149) * _DAY_MS,
            tuning_start_ms=(first_day + 150) * _DAY_MS,
            policy_start_ms=(first_day + 190) * _DAY_MS,
            selection_start_ms=selection_start_ms,
            terminal_start_ms=terminal_start_ms,
            purge_ms=2_000,
            purged_rows=0,
            tuning_early_stop_rows=200,
            tuning_calibration_rows=200,
            tuning_calibration_start_ms=(first_day + 170) * _DAY_MS,
            tuning_internal_purged_rows=0,
        ),
        best_iterations=names,
        probability_calibration={"long": (1.0, 0.0), "short": (1.0, 0.0)},
        threshold_policy=ThresholdPolicy(1.0, 0.6, 0.0),
        policy_search=ThresholdSearchEvidence(400, 200, 200, 20, 1, 1, 0.0, None),
        tuning_auc={"long": 0.5, "short": 0.5},
        tuning_brier={"long": 0.25, "short": 0.25},
        selection_auc={"long": 0.5, "short": 0.5},
        selection_brier={"long": 0.25, "short": 0.25},
        terminal_auc=None,
        terminal_brier=None,
        policy_metrics=empty_metrics,
        selection_metrics=empty_metrics,
        terminal_metrics=None,
        selection_confidence=confidence,
        terminal_confidence=None,
        selection_baselines={"no_trade": empty_metrics},
        terminal_baselines=None,
        model_strings={name: name for name in names},
        deployment_model_strings=None,
        deployment_refit=None,
        dataset_summary=dataset.summary(),
        trained_at="2026-07-10T00:00:00+00:00",
        terminal_evaluated_at=None,
    )
    return dataset, artifact


def _config() -> PrequentialConfig:
    return PrequentialConfig()


def test_fold_plan_is_causal_complete_and_terminal_sealed() -> None:
    dataset, artifact = _dataset_and_artifact()
    plans = plan_prequential_folds(artifact, dataset, _config())
    available = np.maximum(dataset.long_exit_time_ms, dataset.short_exit_time_ms) + int(
        dataset.trade_feature_embargo_ms
    )

    assert len(plans) == 5
    for plan in plans:
        assert np.max(available[plan.training_indexes]) < plan.calibration_start_ms
        assert np.max(available[plan.calibration_indexes]) < plan.policy_start_ms
        assert np.max(available[plan.policy_indexes]) < plan.evaluation_start_ms
        assert np.max(available[plan.evaluation_indexes]) < artifact.split.terminal_start_ms
        assert np.min(dataset.decision_time_ms[plan.evaluation_indexes]) >= plan.evaluation_start_ms
        assert (
            np.max(dataset.decision_time_ms[plan.evaluation_indexes])
            < plan.evaluation_end_exclusive_ms
        )
    evaluated = np.concatenate([plan.evaluation_indexes for plan in plans])
    expected = np.flatnonzero(
        (dataset.decision_time_ms >= artifact.split.selection_start_ms)
        & (dataset.decision_time_ms < artifact.split.terminal_start_ms)
        & (available < artifact.split.terminal_start_ms)
    )
    assert np.array_equal(evaluated, expected)
    assert len(np.unique(evaluated)) == len(evaluated)


def test_fold_plan_preserves_exact_non_midnight_split_boundaries() -> None:
    dataset, artifact = _dataset_and_artifact()
    shifted = replace(
        artifact,
        split=replace(
            artifact.split,
            selection_start_ms=artifact.split.selection_start_ms + 500,
            terminal_start_ms=artifact.split.terminal_start_ms + 500,
        ),
    )
    plans = plan_prequential_folds(shifted, dataset, _config())
    evaluated = np.concatenate([plan.evaluation_indexes for plan in plans])
    available = np.maximum(dataset.long_exit_time_ms, dataset.short_exit_time_ms) + int(
        dataset.trade_feature_embargo_ms
    )
    expected = np.flatnonzero(
        (dataset.decision_time_ms >= shifted.split.selection_start_ms)
        & (dataset.decision_time_ms < shifted.split.terminal_start_ms)
        & (available < shifted.split.terminal_start_ms)
    )

    assert plans[0].evaluation_start_ms == shifted.split.selection_start_ms
    assert plans[-1].evaluation_end_exclusive_ms == shifted.split.terminal_start_ms
    assert np.array_equal(evaluated, expected)


def test_terminal_evaluation_cannot_bypass_prequential_binding() -> None:
    dataset, artifact = _dataset_and_artifact()

    with pytest.raises(ValueError, match="prequential validation evidence"):
        evaluate_microstructure_model_terminal(artifact, dataset)


def test_real_lightgbm_prequential_fold_executes_as_diagnostic(
    tmp_path: Path,
) -> None:
    dataset, artifact = _dataset_and_artifact()
    diagnostic = PrequentialConfig(
        training_window_days=10,
        minimum_training_days=5,
        calibration_days=2,
        policy_days=2,
        evaluation_block_days=2,
        minimum_segment_rows=32,
        minimum_class_rows=16,
        bootstrap_samples=1_000,
        max_folds=1,
    )

    report = evaluate_prequential_microstructure_model(
        artifact,
        dataset,
        config=diagnostic,
        compute_backend="cpu",
        predictions_path=tmp_path / "diagnostic-predictions.csv",
        chart_path=tmp_path / "diagnostic.svg",
        report_path=tmp_path / "diagnostic.json",
    )

    assert report.passed is False
    assert report.coverage["complete_folds"] == 1
    assert report.coverage["failed_folds"] == 0
    assert report.folds[0]["backend_kind"] == "cpu"
    assert len(str(report.folds[0]["model_sha256"])) == 64
    assert "max_folds_truncated_selection_coverage" in report.reasons
    assert "prequential_protocol_deviates_from_locked_promotion_config" in report.reasons


def test_prequential_evaluation_persists_truthful_no_order_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset, artifact = _dataset_and_artifact()

    def fake_fit(_artifact, _dataset, plan, _config, **_kwargs):
        return prequential._FoldModels(
            models={},
            calibration={"long": (1.0, 0.0), "short": (1.0, 0.0)},
            model_sha256=hashlib.sha256(str(plan.fold).encode("ascii")).hexdigest(),
            backend_kind="cpu",
            backend_device="cpu",
            side_training_rows={"long": len(plan.training_indexes), "short": len(plan.training_indexes)},
            side_calibration_rows={
                "long": len(plan.calibration_indexes),
                "short": len(plan.calibration_indexes),
            },
        )

    def fake_predict(_artifact, source, _fitted, indexes):
        long_wins = source.long_net_bps[indexes] > 0.0
        return prequential._PredictionBatch(
            long_edge=np.where(long_wins, 3.0, -1.0),
            short_edge=np.where(long_wins, -1.0, 3.0),
            long_probability=np.where(long_wins, 0.90, 0.10),
            short_probability=np.where(long_wins, 0.10, 0.90),
        )

    monkeypatch.setattr(prequential, "_fit_fold_models", fake_fit)
    monkeypatch.setattr(prequential, "_predict_models", fake_predict)
    predictions = tmp_path / "predictions.csv"
    chart = tmp_path / "prequential.svg"
    report_path = tmp_path / "report.json"

    report = evaluate_prequential_microstructure_model(
        artifact,
        dataset,
        config=_config(),
        compute_backend="cpu",
        predictions_path=predictions,
        chart_path=chart,
        report_path=report_path,
    )

    assert report.passed is True
    assert report.trading_authority is False
    assert report.data_contract["terminal_holdout"] == "not accessed"
    assert report.coverage["selection_coverage_ratio"] == 1.0
    assert report.coverage["failed_folds"] == 0
    assert report.aggregate["metrics"]["total_net_bps"] > 0.0
    assert report.aggregate["profitable_fold_ratio"] == 1.0
    assert all(fold["status"] == "complete" for fold in report.folds)
    assert report.predictions_sha256 == hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert report.chart_sha256 == hashlib.sha256(chart.read_bytes()).hexdigest()
    assert b"\r\n" not in predictions.read_bytes()
    assert b"terminal holdout sealed" in chart.read_bytes()
    attached = attach_verified_prequential_evidence(
        artifact,
        dataset,
        report_path=report_path,
        predictions_path=predictions,
        chart_path=chart,
    )
    assert attached.prequential_validation is not None
    assert attached.prequential_validation.candidate_sha256 == report.candidate_sha256
    assert attached.prequential_validation.report_sha256 == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["coverage"]["evaluated_rows"] -= 1
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="coverage"):
        attach_verified_prequential_evidence(
            artifact,
            dataset,
            report_path=report_path,
            predictions_path=predictions,
            chart_path=chart,
        )
