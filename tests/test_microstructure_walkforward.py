from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simple_ai_trading.microstructure_architecture import GrossActionScoreBatch
from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)
from simple_ai_trading.microstructure_walkforward import (
    WalkForwardFitSpec,
    plan_walk_forward_day,
    recency_weighted_uniqueness,
    select_calibrated_threshold,
    simulate_action_trace,
)


DAY_MS = 86_400_000


def _dataset(days: int = 30, rows_per_day: int = 300):
    rows = days * rows_per_day
    within_day = np.arange(rows_per_day, dtype=np.int64) * 5_000
    times = np.concatenate([day * DAY_MS + within_day for day in range(days)]) + 10_000
    gross = np.resize(np.asarray([20.0, -20.0, 18.0, -18.0]), rows)
    long_net = gross - 12.0
    short_net = -gross - 12.0
    entry_mid = np.full(rows, 100.0)
    exit_mid = entry_mid * np.exp(gross / 10_000.0)
    features = np.ones(
        (rows, len(MICROSTRUCTURE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=300,
        total_latency_ms=750,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=1.0,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=5,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=times,
        long_exit_time_ms=times + 300_750,
        short_exit_time_ms=times + 300_750,
        features=features,
        long_net_bps=long_net,
        short_net_bps=short_net,
        entry_spread_bps=np.full(rows, 2.0),
        exit_spread_bps=np.full(rows, 2.0),
        entry_quote_age_ms=np.full(rows, 10, dtype=np.int64),
        exit_quote_age_ms=np.full(rows, 10, dtype=np.int64),
        entry_bid_price=entry_mid - 0.01,
        entry_ask_price=entry_mid + 0.01,
        fixed_exit_bid_price=exit_mid - 0.01,
        fixed_exit_ask_price=exit_mid + 0.01,
        entry_bid_qty=np.full(rows, 1_000.0),
        entry_ask_qty=np.full(rows, 1_000.0),
        fixed_exit_bid_qty=np.full(rows, 1_000.0),
        fixed_exit_ask_qty=np.full(rows, 1_000.0),
        long_l1_participation=np.full(rows, 0.01),
        short_l1_participation=np.full(rows, 0.01),
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
        source_evidence=None,
    )
    return dataset, gross


def test_daily_plan_is_contiguous_purged_and_prior_only() -> None:
    dataset, _gross = _dataset()
    spec = WalkForwardFitSpec("rolling", 10, 2, 2, None)
    plan = plan_walk_forward_day(
        dataset,
        np.ones(dataset.rows, dtype=bool),
        evaluation_day_id=20,
        corpus_start_day_id=0,
        spec=spec,
    )

    assert plan.evidence["train_start_day_id"] == 6
    assert plan.evidence["early_stop_start_day_id"] == 16
    assert plan.evidence["calibration_start_day_id"] == 18
    assert plan.evidence["evaluation_start_day_id"] == 20
    assert np.max(dataset.long_exit_time_ms[plan.train_indexes]) < 16 * DAY_MS
    assert np.max(dataset.long_exit_time_ms[plan.early_stop_indexes]) < 18 * DAY_MS
    assert np.max(dataset.long_exit_time_ms[plan.calibration_indexes]) < 20 * DAY_MS
    assert np.min(dataset.decision_time_ms[plan.evaluation_indexes]) >= 20 * DAY_MS


def test_expanding_plan_and_recency_weights_are_causal() -> None:
    dataset, _gross = _dataset()
    spec = WalkForwardFitSpec("expanding", 0, 2, 2, 3.0)
    plan = plan_walk_forward_day(
        dataset,
        np.ones(dataset.rows, dtype=bool),
        evaluation_day_id=20,
        corpus_start_day_id=0,
        spec=spec,
    )
    indexes = plan.train_indexes
    base = np.ones(len(indexes), dtype=np.float32)
    weights = recency_weighted_uniqueness(
        dataset,
        indexes,
        base,
        half_life_days=3.0,
    )

    assert plan.evidence["train_start_day_id"] == 0
    assert np.mean(weights) == pytest.approx(1.0)
    assert weights[-1] > weights[0]
    np.testing.assert_array_equal(
        recency_weighted_uniqueness(
            dataset,
            indexes,
            base,
            half_life_days=None,
        ),
        base,
    )


def test_trace_is_non_overlapping_and_flat_by_utc_day() -> None:
    dataset, gross = _dataset(days=2, rows_per_day=100)
    endpoints = np.arange(dataset.rows, dtype=np.int64)
    score = GrossActionScoreBatch(
        endpoint_indexes=endpoints,
        side=np.where(gross > 0.0, 1, -1).astype(np.int8),
        strength=np.full(dataset.rows, 1.0),
        method="direction_magnitude",
        strength_units="confidence_weighted_basis_points",
    )
    trace = simulate_action_trace(dataset, gross, score, strength_threshold=0.5)

    assert trace.metrics.trades > 0
    assert trace.metrics.active_days == 2
    assert trace.metrics.total_net_bps > 0.0
    assert all(
        right - left >= 300_750 or right // DAY_MS != left // DAY_MS
        for left, right in zip(
            trace.timestamps_ms,
            trace.timestamps_ms[1:],
        )
    )
    assert trace.portfolio_claim is False
    assert trace.trading_authority is False


def test_threshold_selection_can_accept_or_force_abstention() -> None:
    dataset, gross = _dataset(days=2, rows_per_day=100)
    endpoints = np.arange(dataset.rows, dtype=np.int64)
    profitable = GrossActionScoreBatch(
        endpoint_indexes=endpoints,
        side=np.where(gross > 0.0, 1, -1).astype(np.int8),
        strength=np.linspace(0.1, 1.0, dataset.rows),
        method="direction_magnitude",
        strength_units="confidence_weighted_basis_points",
    )
    accepted = select_calibrated_threshold(
        dataset,
        gross,
        profitable,
        quantiles=(0.5, 0.8),
        minimum_trades=2,
        maximum_drawdown_bps=50.0,
        minimum_positive_day_ratio=0.5,
        drawdown_penalty=0.25,
    )
    rejected = select_calibrated_threshold(
        replace(
            dataset,
            long_net_bps=-np.abs(dataset.long_net_bps),
            short_net_bps=-np.abs(dataset.short_net_bps),
        ),
        gross,
        profitable,
        quantiles=(0.5, 0.8),
        minimum_trades=2,
        maximum_drawdown_bps=50.0,
        minimum_positive_day_ratio=0.5,
        drawdown_penalty=0.25,
    )

    assert accepted.accepted is True
    assert accepted.threshold is not None
    assert accepted.selected_trace.metrics.total_net_bps > 0.0
    assert rejected.accepted is False
    assert rejected.threshold is None
    assert rejected.selected_trace.metrics.trades == 0
    assert rejected.rejection_reasons == ("no_calibration_threshold_passed_risk_gates",)


def test_trace_rejects_unsorted_or_forged_scores() -> None:
    dataset, gross = _dataset(days=2, rows_per_day=100)
    score = GrossActionScoreBatch(
        endpoint_indexes=np.asarray([1, 0], dtype=np.int64),
        side=np.asarray([1, -1], dtype=np.int8),
        strength=np.ones(2),
        method="direction_magnitude",
        strength_units="confidence_weighted_basis_points",
    )
    with pytest.raises(ValueError, match="trace contract"):
        simulate_action_trace(dataset, gross, score, strength_threshold=0.5)

    forged = replace(
        score,
        endpoint_indexes=np.asarray([0, 1], dtype=np.int64),
        side=np.asarray([0, 1], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="trace contract"):
        simulate_action_trace(dataset, gross, forged, strength_threshold=0.5)


@pytest.mark.parametrize(
    "values",
    [
        ("", 10, 2, 2, None),
        ("bad-window", 5, 2, 2, None),
        ("bad-stop", 10, 0, 2, None),
        ("bad-calibration", 10, 2, 0, None),
        ("bad-half-life", 10, 2, 2, 0.5),
    ],
)
def test_walk_forward_spec_rejects_invalid_contract(values) -> None:
    with pytest.raises(ValueError, match="walk-forward"):
        WalkForwardFitSpec(*values)
