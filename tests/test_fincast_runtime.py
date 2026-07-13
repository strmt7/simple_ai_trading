from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.fincast_runtime import (
    build_fincast_causal_state_series,
    fincast_context_valid_mask,
)


def test_causal_state_series_holds_only_past_observations_with_bounded_age() -> None:
    seconds, prices, evidence = build_fincast_causal_state_series(
        observed_second_ms=np.asarray([0, 2_000, 9_000], dtype=np.int64),
        observed_close_mid=np.asarray([100.0, 102.0, 110.0], dtype=np.float32),
        target_start_ms=0,
        target_end_ms=10_000,
    )

    assert seconds.tolist() == [
        0,
        1_000,
        2_000,
        3_000,
        4_000,
        5_000,
        6_000,
        7_000,
        9_000,
        10_000,
    ]
    assert prices.tolist() == [
        100.0,
        100.0,
        102.0,
        102.0,
        102.0,
        102.0,
        102.0,
        102.0,
        110.0,
        110.0,
    ]
    assert evidence.exact_observation_rows == 3
    assert evidence.carried_state_rows == 7
    assert evidence.expired_state_rows == 1
    assert evidence.maximum_retained_state_age_seconds == 5


def test_causal_state_series_can_seed_target_from_prior_observation() -> None:
    seconds, prices, _evidence = build_fincast_causal_state_series(
        observed_second_ms=np.asarray([0, 7_000], dtype=np.int64),
        observed_close_mid=np.asarray([100.0, 107.0], dtype=np.float32),
        target_start_ms=3_000,
        target_end_ms=8_000,
    )

    assert seconds.tolist() == [3_000, 4_000, 5_000, 7_000, 8_000]
    assert prices.tolist() == [100.0, 100.0, 100.0, 107.0, 107.0]


def test_context_mask_rejects_missing_anchor_and_interrupted_window() -> None:
    seconds = np.delete(
        np.arange(800, dtype=np.int64) * 1_000,
        100,
    )
    decisions = np.asarray([101_000, 512_000, 700_000], dtype=np.int64)

    valid = fincast_context_valid_mask(
        second_ms=seconds,
        decision_time_ms=decisions,
    )

    assert valid.tolist() == [False, False, True]


def test_context_mask_rejects_non_monotonic_source() -> None:
    with pytest.raises(ValueError, match="validity source"):
        fincast_context_valid_mask(
            second_ms=np.asarray([0, 1_000, 1_000], dtype=np.int64),
            decision_time_ms=np.asarray([2_000], dtype=np.int64),
        )
