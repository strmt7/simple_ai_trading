from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.barrier_competing_risk_analysis import (
    replay_fixed_trades,
    select_fixed_policy_trades,
)
from simple_ai_trading.barrier_competing_risk_tcn_model import (
    EVENT_CLASSES,
    BarrierCompetingRiskForecastBundle,
    barrier_event_classes,
    barrier_risk_targets,
    cpu_barrier_competing_risk_preflight,
    fit_barrier_target_baselines,
    fit_event_temperature,
)
from simple_ai_trading.barrier_payoff_data import (
    STOP_EVENT,
    TAKE_PROFIT_EVENT,
    TIMEOUT_EVENT,
    BarrierPayoffDataset,
    BarrierSpecification,
)
from simple_ai_trading.cross_asset_cost_data import SYMBOLS
from simple_ai_trading.minute_logistic_mixture_tcn_model import (
    FEATURE_COUNT,
    MinuteTemporalDataset,
    RobustFeatureScaler,
)


def _barrier_dataset(timestamps: int = EVENT_CLASSES) -> BarrierPayoffDataset:
    shape = (timestamps, len(SYMBOLS), 2)
    event_code = np.empty(shape, dtype=np.int8)
    event_minute = np.empty(shape, dtype=np.int16)
    base_codes = np.concatenate(
        (
            np.full(60, STOP_EVENT, dtype=np.int8),
            np.full(60, TAKE_PROFIT_EVENT, dtype=np.int8),
            np.asarray([TIMEOUT_EVENT], dtype=np.int8),
        )
    )
    base_minutes = np.concatenate(
        (
            np.arange(1, 61, dtype=np.int16),
            np.arange(1, 61, dtype=np.int16),
            np.asarray([60], dtype=np.int16),
        )
    )
    for symbol_index in range(len(SYMBOLS)):
        for side_index in range(2):
            event_code[:, symbol_index, side_index] = np.resize(base_codes, timestamps)
            event_minute[:, symbol_index, side_index] = np.resize(
                base_minutes, timestamps
            )
    stop = np.full((timestamps, len(SYMBOLS)), 50.0, dtype=np.float32)
    take = np.full_like(stop, 100.0)
    net = np.empty(shape, dtype=np.float32)
    net[event_code == STOP_EVENT] = -62.0
    net[event_code == TAKE_PROFIT_EVENT] = 88.0
    net[event_code == TIMEOUT_EVENT] = 13.0
    for symbol_index in (0, 2):
        timeout = event_code[:, symbol_index] == TIMEOUT_EVENT
        net[:, symbol_index][timeout] = -13.0
    roles = {
        "training": np.ones(timestamps, dtype=bool),
        "early_stop": np.zeros(timestamps, dtype=bool),
        "calibration": np.zeros(timestamps, dtype=bool),
        "viability": np.ones(timestamps, dtype=bool),
    }
    return BarrierPayoffDataset(
        timestamps_ms=np.arange(timestamps, dtype=np.int64) * 300_000,
        stop_bps=stop,
        take_profit_bps=take,
        event_code=event_code,
        event_minute=event_minute,
        price_return_bps=net + 12.0,
        funding_cash_flow_bps=np.zeros(shape, dtype=np.float32),
        net_payoff_bps=net,
        gap_through_slippage_bps=np.zeros(shape, dtype=np.float32),
        ambiguous_stop_first=np.zeros(shape, dtype=bool),
        role_masks=roles,
        specification=BarrierSpecification(
            horizon_minutes=60,
            stop_volatility_multiple=1.0,
            take_profit_to_stop_ratio=2.0,
            minimum_stop_bps=24.0,
            maximum_stop_bps=80.0,
            round_trip_execution_charge_bps=12.0,
        ),
        dataset_sha256="a" * 64,
    )


def _temporal(timestamps: int) -> MinuteTemporalDataset:
    return MinuteTemporalDataset(
        feature_names=tuple(f"feature-{index}" for index in range(FEATURE_COUNT)),
        timestamps_ms=np.arange(timestamps, dtype=np.int64) * 300_000,
        features=np.zeros((timestamps, len(SYMBOLS), FEATURE_COUNT), dtype=np.float32),
        signed_target_bps=np.zeros((timestamps, len(SYMBOLS), 4), dtype=np.float32),
        role_masks={},
        feature_stream_sha256="b" * 64,
        target_stream_sha256="c" * 64,
        dataset_sha256="d" * 64,
        source_evidence={},
    )


def test_exact_event_classes_and_training_residual_baselines() -> None:
    dataset = _barrier_dataset()
    classes = barrier_event_classes(dataset)
    targets = barrier_risk_targets(dataset)
    baselines = fit_barrier_target_baselines(dataset, classes, targets)

    assert classes[:, 0, 0].tolist() == list(range(EVENT_CLASSES))
    assert np.allclose(baselines.stop_residual_mean_risk_units, 0.0)
    assert np.allclose(baselines.take_residual_mean_risk_units, 0.0)
    assert np.allclose(baselines.event_class_probability, 1.0 / EVENT_CLASSES)
    assert targets[0, 0, 0] == pytest.approx(-1.24)
    assert targets[60, 0, 0] == pytest.approx(1.76)


def test_event_temperature_is_bounded_and_nonworsening() -> None:
    labels = np.arange(EVENT_CLASSES, dtype=np.int64)
    logits = np.zeros((EVENT_CLASSES, EVENT_CLASSES), dtype=np.float64)
    logits[np.arange(EVENT_CLASSES), labels] = 3.0

    calibration = fit_event_temperature(logits, labels)

    assert 0.5 <= calibration.temperature <= 3.0
    assert (
        calibration.multinomial_log_loss_after
        <= calibration.multinomial_log_loss_before
    )


def test_cpu_preflight_updates_all_financial_heads_without_warnings() -> None:
    _, report = cpu_barrier_competing_risk_preflight()

    assert report["cpu_fallback_warnings"] == 0
    assert [item["candidate_id"] for item in report["candidates"]] == [
        "direct_barrier_mean_tcn",
        "competing_risk_barrier_tcn",
    ]
    for candidate in report["candidates"]:
        assert min(candidate["parameter_changes"].values()) > 0.0


def test_fixed_policy_does_not_overlap_a_still_open_symbol_position() -> None:
    timestamps = 4
    barrier = _barrier_dataset(timestamps)
    barrier.event_code[:, 0, 1] = TIMEOUT_EVENT
    barrier.event_minute[:, 0, 1] = 60
    temporal = _temporal(timestamps)
    classes = barrier_event_classes(_barrier_dataset())
    full = _barrier_dataset()
    baselines = fit_barrier_target_baselines(full, classes, barrier_risk_targets(full))
    actions = np.full((3, timestamps, len(SYMBOLS), 2), -1.0, dtype=np.float32)
    actions[:, :, 0, 1] = 5.0
    bundle = BarrierCompetingRiskForecastBundle(
        candidate_id="direct_barrier_mean_tcn",
        global_indices=np.arange(timestamps, dtype=np.int64),
        seed_event_true_probabilities=np.ones_like(actions),
        seed_event_probability_square_sums=np.ones_like(actions),
        seed_event_group_probabilities=np.zeros(
            (3, timestamps, len(SYMBOLS), 2, 3), dtype=np.float32
        ),
        seed_event_expected_minutes=np.full_like(actions, 60.0),
        seed_timeout_profit_probabilities=np.full_like(actions, 0.5),
        seed_timeout_mean_risk_units=np.zeros_like(actions),
        seed_action_values_bps=actions,
        artifacts=(),
        feature_scaler=RobustFeatureScaler(
            median=np.zeros(FEATURE_COUNT), scaled_iqr=np.ones(FEATURE_COUNT)
        ),
        target_baselines=baselines,
        backend_kind="cpu",
        backend_device="cpu",
        training_history=(),
    )

    trades = select_fixed_policy_trades(bundle, temporal, barrier)
    replay = replay_fixed_trades(trades, temporal, barrier, candidate_index=0)

    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTCUSDT"
    assert replay["scenarios"]["base"]["closed_trades"] == 1
    assert replay["scenarios"]["stress"]["mean_net_payoff_bps"] == pytest.approx(
        float(trades[0]["base_net_payoff_bps"]) - 4.0
    )
