from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest

from simple_ai_trading.derivatives_hurdle_data import DerivativesSourceEvidence
from simple_ai_trading.stateful_turnover_model import (
    AI_FACTOR_NAMES,
    StatefulHourlyDataset,
    build_ai_factor_matrix,
    replay_independent_hourly,
    replay_stateful_policy,
)


def _timestamp_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def _dataset(hours: int, *, target_bps: float = 10.0) -> StatefulHourlyDataset:
    timestamps = _timestamp_ms("2025-01-01T00:00:00") + np.arange(hours) * 3_600_000
    decision_time_ms = np.repeat(timestamps, 3).astype(np.int64)
    symbol_index = np.tile(np.arange(3, dtype=np.int8), hours)
    baseline = np.zeros((hours * 3, 1), dtype=np.float32)
    return StatefulHourlyDataset(
        feature_names=("fixture",),
        baseline_features=baseline,
        augmented_features=baseline.copy(),
        decision_time_ms=decision_time_ms,
        symbol_index=symbol_index,
        signed_pre_transition_utility_bps=np.full(
            hours * 3, target_bps, dtype=np.float32
        ),
        funding_cash_flow_bps=np.zeros(hours * 3, dtype=np.float32),
        source_evidence=cast(DerivativesSourceEvidence, None),
        dataset_sha256="fixture",
    )


def _predictions(hours: int, values: list[float]) -> np.ndarray:
    if len(values) != hours:
        raise ValueError("fixture prediction length differs")
    return np.repeat(np.asarray(values, dtype=np.float32), 3)


def test_ai_factor_programs_match_frozen_signs_and_floors() -> None:
    names = (
        "target_return_60m_bps",
        "target_path_efficiency_60m",
        "target_return_15m_bps",
        "target_signed_taker_flow_15m",
        "target_return_zscore_240m",
        "target_beta_residual_return_60m_bps",
        "target_realized_volatility_60m_bps",
        "target_signed_taker_flow_5m",
        "target_same_minute_of_week_liquidity_ratio",
        "target_upside_semivolatility_60m_bps",
        "target_downside_semivolatility_60m_bps",
    )
    values = np.asarray(
        [[50.0, 0.5, 25.0, 0.2, 1.0, 10.0, 0.0, 0.4, 3.0, 3.0, 1.0]],
        dtype=np.float32,
    )

    actual = build_ai_factor_matrix(values, names)

    expected = np.asarray(
        [
            np.tanh(1.0) * 0.5,
            np.tanh(1.0) * 0.2,
            -np.tanh(1.0) * 0.5,
            2.0,
            0.8,
            0.5,
        ],
        dtype=np.float32,
    )
    assert actual.shape == (1, len(AI_FACTOR_NAMES))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-7)


def test_ai_factor_programs_fail_closed_on_missing_or_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_ai_factor_matrix(np.zeros((2, 1), dtype=np.float32), ("unrelated",))
    names = (
        "target_return_60m_bps",
        "target_path_efficiency_60m",
        "target_return_15m_bps",
        "target_signed_taker_flow_15m",
        "target_return_zscore_240m",
        "target_beta_residual_return_60m_bps",
        "target_realized_volatility_60m_bps",
        "target_signed_taker_flow_5m",
        "target_same_minute_of_week_liquidity_ratio",
        "target_upside_semivolatility_60m_bps",
        "target_downside_semivolatility_60m_bps",
    )
    values = np.zeros((1, len(names)), dtype=np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        build_ai_factor_matrix(values, names)


def test_stateful_policy_charges_only_entry_and_final_exit_when_signal_persists() -> (
    None
):
    dataset = _dataset(3)
    result = replay_stateful_policy(
        dataset,
        _predictions(3, [13.0, 13.0, 13.0]),
        feature_set="baseline_71",
        mode="long_only",
        cost_scenario="base",
        cost_bps=6.0,
        seed=1,
    )

    np.testing.assert_allclose(result.portfolio_return_bps, [4.0, 10.0, 4.0])
    np.testing.assert_array_equal(result.positions, np.ones((3, 3), dtype=np.int8))
    assert result.metrics["transition_events"] == 6
    assert result.metrics["transition_units"] == 6.0
    assert result.metrics["entries"] == 3
    assert result.metrics["final_boundary_exits"] == 3


def test_independent_hourly_comparator_recharges_the_round_trip_each_hour() -> None:
    dataset = _dataset(3)
    result = replay_independent_hourly(
        dataset,
        _predictions(3, [13.0, 13.0, 13.0]),
        feature_set="baseline_71",
        mode="long_only",
        cost_scenario="base",
        cost_bps=6.0,
        seed=2,
    )

    np.testing.assert_allclose(result.portfolio_return_bps, [-2.0, -2.0, -2.0])
    assert result.metrics["transition_units"] == 18.0


def test_cost_hurdle_is_strict_and_long_short_reversal_pays_two_units() -> None:
    dataset = _dataset(2, target_bps=0.0)
    flat = replay_stateful_policy(
        dataset,
        _predictions(2, [12.0, 12.0]),
        feature_set="baseline_71",
        mode="long_only",
        cost_scenario="base",
        cost_bps=6.0,
        seed=3,
    )
    assert not np.any(flat.positions)
    assert flat.metrics["transition_units"] == 0.0

    reversal = replay_stateful_policy(
        dataset,
        _predictions(2, [13.0, -25.0]),
        feature_set="baseline_71",
        mode="long_short",
        cost_scenario="base",
        cost_bps=6.0,
        seed=4,
    )
    np.testing.assert_array_equal(reversal.positions[:, 0], [1, -1])
    np.testing.assert_allclose(reversal.transition_units[:, 0], [1.0, 3.0])
    assert reversal.metrics["reversals"] == 3
    assert reversal.metrics["final_boundary_exits"] == 3


def test_maximum_hold_forces_one_flat_interval_before_reentry() -> None:
    dataset = _dataset(26, target_bps=0.0)
    result = replay_stateful_policy(
        dataset,
        _predictions(26, [13.0] * 26),
        feature_set="baseline_71",
        mode="long_only",
        cost_scenario="base",
        cost_bps=6.0,
        seed=5,
    )

    np.testing.assert_array_equal(result.positions[23], [1, 1, 1])
    np.testing.assert_array_equal(result.position_age_hours[23], [24, 24, 24])
    np.testing.assert_array_equal(result.positions[24], [0, 0, 0])
    np.testing.assert_array_equal(result.positions[25], [1, 1, 1])
    assert result.metrics["maximum_hold_forced_exits"] == 3
    assert result.metrics["maximum_completed_holding_hours"] == 24
