from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from simple_ai_trading.derivatives_hurdle_data import DerivativesHurdleDataset
from simple_ai_trading.rolling_refit_model import (
    _replay_candidate_mask,
    _training_weights,
    _window_mask,
    frozen_monthly_schedules,
)


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def _dataset(
    decision_time_ms: np.ndarray,
    symbol_index: np.ndarray,
    *,
    long_utility: np.ndarray | None = None,
    short_utility: np.ndarray | None = None,
) -> DerivativesHurdleDataset:
    rows = decision_time_ms.size
    long_values = (
        np.asarray(long_utility, dtype=np.float32)
        if long_utility is not None
        else np.full(rows, 10.0, dtype=np.float32)
    )
    short_values = (
        np.asarray(short_utility, dtype=np.float32)
        if short_utility is not None
        else np.full(rows, -5.0, dtype=np.float32)
    )
    return DerivativesHurdleDataset(
        feature_names=("x",),
        price_flow_feature_count=1,
        features=np.zeros((rows, 1), dtype=np.float32),
        decision_time_ms=decision_time_ms,
        symbol_index=symbol_index,
        target_class={120: np.where(long_values > 0.0, 2, 1).astype(np.int8)},
        long_net_utility_bps={120: long_values},
        short_net_utility_bps={120: short_values},
        funding_cash_flow_bps={120: np.zeros(rows, dtype=np.float32)},
        role_masks={},
        source_evidence=None,  # type: ignore[arg-type]
        source_exclusions={},
    )


def test_frozen_monthly_schedule_has_exact_first_and_last_roles() -> None:
    schedules = frozen_monthly_schedules()

    assert len(schedules) == 6
    assert schedules[0].asdict() == {
        "evaluation_month": "2025-01",
        "training_start": "2022-10-01",
        "training_end": "2024-09-30",
        "early_stop_start": "2024-10-01",
        "early_stop_end": "2024-11-30",
        "calibration_start": "2024-12-01",
        "calibration_end": "2024-12-31",
        "evaluation_start": "2025-01-01",
        "evaluation_end": "2025-01-31",
    }
    assert schedules[-1].training_start == "2023-03-01"
    assert schedules[-1].training_end == "2025-02-28"
    assert schedules[-1].evaluation_start == "2025-06-01"
    assert schedules[-1].evaluation_end == "2025-06-30"


def test_window_mask_embargoes_targets_that_cross_role_boundary() -> None:
    dataset = _dataset(
        np.asarray(
            [
                _ms("2024-09-30T21:58:00"),
                _ms("2024-09-30T21:59:00"),
                _ms("2024-10-01T00:00:00"),
            ],
            dtype=np.int64,
        ),
        np.zeros(3, dtype=np.int8),
    )
    mask = _window_mask(
        dataset,
        horizon_minutes=120,
        start="2024-09-01",
        end="2024-09-30",
    )

    assert mask.tolist() == [True, False, False]


def test_bounded_utility_weights_are_training_only_and_capped() -> None:
    dataset = _dataset(
        np.arange(12, dtype=np.int64),
        np.zeros(12, dtype=np.int8),
        long_utility=np.asarray([0.0, *([10.0] * 9), 1000.0, 1_000_000.0]),
        short_utility=np.full(12, -1.0),
    )
    training = np.asarray([*([True] * 11), False])
    weights, normalizer = _training_weights(
        dataset,
        horizon_minutes=120,
        training_mask=training,
        weighting="bounded_economic_utility",
        target_head="direct",
    )

    assert normalizer is not None
    assert 20.0 < normalizer < 1000.0
    assert weights.shape == (11,)
    assert weights[0] == 1.0
    assert float(np.min(weights)) >= 1.0
    assert float(np.max(weights)) == 3.0


def test_authoritative_replay_prevents_overlap_across_month_boundary() -> None:
    dataset = _dataset(
        np.asarray(
            [
                _ms("2025-01-31T23:55:00"),
                _ms("2025-02-01T00:00:00"),
                _ms("2025-02-01T03:00:00"),
            ],
            dtype=np.int64,
        ),
        np.zeros(3, dtype=np.int8),
        long_utility=np.asarray([10.0, 20.0, 30.0]),
    )
    outcome = _replay_candidate_mask(
        dataset,
        candidate_mask=np.ones(3, dtype=bool),
        direction=np.ones(3, dtype=np.int8),
        horizon_minutes=120,
        period_start="2025-01-01",
        period_end="2025-06-30",
        maximum_action_probability=None,
        direction_probability_margin=None,
        bootstrap_samples=0,
        bootstrap_seed=1,
    )

    assert outcome.metrics.total_trades == 2
    assert outcome.metrics.overlap_rejections == 1
    assert outcome.net_return_bps.tolist() == [10.0, 30.0]
    assert outcome.metrics.trades_by_symbol == {
        "BTCUSDT": 2,
        "ETHUSDT": 0,
        "SOLUSDT": 0,
    }
