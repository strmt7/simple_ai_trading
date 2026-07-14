from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.bounded_alpha_lightgbm import (
    BoundedAlphaSpec,
    VIEW_IDS,
    build_trade_plan,
    consensus_decisions,
    load_bounded_alpha_model,
    predict_bounded_alpha_model,
    replay_trade_plan,
    save_bounded_alpha_model,
    train_bounded_alpha_model,
)
from simple_ai_trading.cross_asset_cost_data import MINUTE_MS, SYMBOLS
from simple_ai_trading.stop_time_payoff_data import (
    TIMEOUT_EVENT,
    StopTimePayoffDataset,
    StopTimeSpecification,
)


FEATURE_NAMES = (
    "signal",
    "target_same_minute_of_week_liquidity_ratio",
    "target_quote_volume_vs_1440m_mean",
    "target_realized_volatility_60m_bps",
    "target_realized_volatility_1440m_bps",
)


def _model_source() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(55)
    timestamps = np.arange(900, dtype=np.int64) * 60 * MINUTE_MS
    features = rng.normal(size=(900, len(SYMBOLS), len(FEATURE_NAMES))).astype(
        np.float32
    )
    features[..., 1] = 1.0
    features[..., 2] = 1.0
    features[..., 3] = 4.0
    features[..., 4] = 5.0
    stop = np.full((900, len(SYMBOLS)), 80.0, dtype=np.float32)
    noise = rng.normal(scale=2.0, size=(900, len(SYMBOLS)))
    long_target = 12.0 * features[..., 0] + noise
    short_target = -12.0 * features[..., 0] + noise
    exit_time = np.broadcast_to(
        timestamps[:, None] + 61 * MINUTE_MS,
        stop.shape,
    ).copy()
    return timestamps, features, stop, long_target, short_target, exit_time


def test_bounded_alpha_model_refits_reloads_and_predicts(tmp_path: Path) -> None:
    timestamps, features, stop, long_target, short_target, exit_time = _model_source()
    roles = {
        "iteration_training": np.arange(900) < 400,
        "iteration_selection": (np.arange(900) >= 402) & (np.arange(900) < 600),
        "final_refit": np.arange(900) < 700,
        "policy_development": np.arange(900) >= 702,
    }
    model = train_bounded_alpha_model(
        treatment_id="baseline_71",
        view_id="raw_uniform",
        seed=5501,
        features=features,
        feature_names=FEATURE_NAMES,
        timestamps_ms=timestamps,
        stop_bps=stop,
        long_target_bps=long_target,
        short_target_bps=short_target,
        role_masks=roles,
        long_exit_time_ms=exit_time,
        short_exit_time_ms=exit_time,
        source_dataset_sha256="a" * 64,
        payoff_dataset_sha256="b" * 64,
        spec=BoundedAlphaSpec(
            min_data_in_leaf=32,
            maximum_boosting_rounds=20,
            early_stopping_rounds=5,
        ),
        compute_backend="cpu",
    )
    prediction = predict_bounded_alpha_model(model, features[-10:], stop[-10:])
    assert prediction.shape == (10, len(SYMBOLS), 2)
    assert model.iteration_selection_mae_skill["long"] > 0.0
    assert model.iteration_selection_mae_skill["short"] > 0.0
    path = tmp_path / "model.json"
    save_bounded_alpha_model(path, model)
    loaded = load_bounded_alpha_model(path)
    reloaded = predict_bounded_alpha_model(loaded, features[-10:], stop[-10:])
    np.testing.assert_allclose(prediction, reloaded, rtol=0.0, atol=1e-12)


def _payoff(timestamps: int) -> StopTimePayoffDataset:
    shape = (timestamps, len(SYMBOLS))
    decision_time = np.arange(timestamps, dtype=np.int64) * 60 * MINUTE_MS
    exit_time = np.broadcast_to(
        decision_time[:, None] + 61 * MINUTE_MS,
        shape,
    ).copy()
    zeros = np.zeros(shape, dtype=np.float32)
    events = np.full(shape, TIMEOUT_EVENT, dtype=np.int8)
    minutes = np.full(shape, 60, dtype=np.int16)
    return StopTimePayoffDataset(
        timestamps_ms=decision_time,
        stop_bps=np.full(shape, 100.0, dtype=np.float32),
        long_event_code=events.copy(),
        short_event_code=events.copy(),
        long_event_minute=minutes.copy(),
        short_event_minute=minutes.copy(),
        long_exit_time_ms=exit_time.copy(),
        short_exit_time_ms=exit_time.copy(),
        long_price_return_bps=np.full(shape, 40.0, dtype=np.float32),
        short_price_return_bps=np.full(shape, -40.0, dtype=np.float32),
        long_funding_cash_flow_bps=zeros.copy(),
        short_funding_cash_flow_bps=zeros.copy(),
        long_net_payoff_bps=np.full(shape, 24.0, dtype=np.float32),
        short_net_payoff_bps=np.full(shape, -56.0, dtype=np.float32),
        long_gap_through_slippage_bps=zeros.copy(),
        short_gap_through_slippage_bps=zeros.copy(),
        source_dataset_sha256="a" * 64,
        specification=StopTimeSpecification(
            horizon_minutes=60,
            stop_volatility_multiple=1.5,
            minimum_stop_bps=40.0,
            maximum_stop_bps=250.0,
            round_trip_execution_charge_bps=16.0,
        ),
        dataset_sha256="b" * 64,
    )


def test_consensus_and_replay_apply_market_and_portfolio_gates() -> None:
    timestamps = 30
    features = np.ones((timestamps, len(SYMBOLS), len(FEATURE_NAMES)))
    features[..., 3] = 4.0
    features[..., 4] = 5.0
    predictions = np.empty((len(VIEW_IDS), 3, timestamps, len(SYMBOLS), 2))
    predictions[..., 0] = 20.0
    predictions[..., 1] = -10.0
    features[0, 0, 1] = 0.1
    decisions = consensus_decisions(predictions, features, FEATURE_NAMES)
    assert decisions.actions[0, 0] == 0
    assert np.all(decisions.actions[1:] == 1)

    payoff = _payoff(timestamps)
    interval = np.ones(timestamps, dtype=bool)
    plan = build_trade_plan(payoff, decisions, interval)
    assert plan.closed_trades == timestamps * len(SYMBOLS) - 1
    first_batch = plan.size_fraction[plan.decision_index == 0]
    assert np.sum(first_batch) <= (1.0 / 3.0) + 1e-12
    stress = replay_trade_plan(
        payoff,
        plan,
        interval,
        scenario="stress",
        round_trip_execution_charge_bps=16.0,
    )
    base = replay_trade_plan(
        payoff,
        plan,
        interval,
        scenario="base",
        round_trip_execution_charge_bps=12.0,
    )
    assert stress.metrics["closed_trades"] == plan.closed_trades
    assert stress.metrics["maximum_position_fraction"] <= 1.0 / 3.0
    assert base.metrics["total_return_fraction"] > stress.metrics["total_return_fraction"]
    assert stress.metrics["profit_reinvestment"] is False


def test_consensus_requires_two_positive_seeds_in_every_view() -> None:
    features = np.ones((2, len(SYMBOLS), len(FEATURE_NAMES)))
    features[..., 3] = 4.0
    features[..., 4] = 5.0
    predictions = np.empty((len(VIEW_IDS), 3, 2, len(SYMBOLS), 2))
    predictions[..., 0] = 10.0
    predictions[..., 1] = -10.0
    predictions[0, :2, ..., 0] = -1.0
    decisions = consensus_decisions(predictions, features, FEATURE_NAMES)
    assert not np.any(decisions.actions)


def test_model_rejects_label_boundary_crossing() -> None:
    timestamps, features, stop, long_target, short_target, exit_time = _model_source()
    roles = {
        "iteration_training": np.arange(900) < 400,
        "iteration_selection": (np.arange(900) >= 400) & (np.arange(900) < 600),
        "final_refit": np.arange(900) < 700,
        "policy_development": np.arange(900) >= 700,
    }
    with pytest.raises(ValueError, match="cross a chronological boundary"):
        train_bounded_alpha_model(
            treatment_id="baseline_71",
            view_id="raw_uniform",
            seed=5501,
            features=features,
            feature_names=FEATURE_NAMES,
            timestamps_ms=timestamps,
            stop_bps=stop,
            long_target_bps=long_target,
            short_target_bps=short_target,
            role_masks=roles,
            long_exit_time_ms=exit_time,
            short_exit_time_ms=exit_time,
            source_dataset_sha256="a" * 64,
            payoff_dataset_sha256="b" * 64,
            spec=BoundedAlphaSpec(
                min_data_in_leaf=32,
                maximum_boosting_rounds=20,
                early_stopping_rounds=5,
            ),
            compute_backend="cpu",
        )
