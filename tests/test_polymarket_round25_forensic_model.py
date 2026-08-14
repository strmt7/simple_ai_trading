from __future__ import annotations

import numpy as np

from simple_ai_trading.polymarket_round25_forensic_model import (
    _Rows,
    _fit_logistic,
    _metrics,
    _predict_logistic,
    _weighted_isotonic,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
)


def _rows() -> _Rows:
    condition_ids = tuple(
        f"0x{condition + 1:064x}"
        for condition in range(4)
        for _ in range(16)
    )
    labels = np.repeat(np.asarray([0.0, 1.0, 0.0, 1.0]), 16)
    feature = np.zeros((64, len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)))
    feature[:, 0] = labels * 2.0 - 1.0
    return _Rows(
        role="train",
        condition_ids=condition_ids,
        event_start_ms=np.repeat(np.arange(4, dtype=np.int64) * 300_000 + 300_000, 16),
        decision_time_ms=np.repeat(np.arange(4, dtype=np.int64) * 300_000 + 300_000, 16)
        + np.tile(np.arange(16, dtype=np.int64), 4),
        features=feature,
        prior=np.full(64, 0.5),
        labels=labels,
        source_sha256=("a" * 64,) * 64,
    ).validated(require_labels=True)


def test_logistic_residual_improves_a_real_signal_without_target_shortcuts() -> None:
    rows = _rows()
    intercept, coefficients = _fit_logistic(rows, rows.features, l2=0.1)
    probability = _predict_logistic(rows, rows.features, intercept, coefficients)

    assert _metrics(rows, probability)["condition_equal_log_loss"] < _metrics(
        rows, rows.prior
    )["condition_equal_log_loss"]
    assert coefficients[0] > 0.0


def test_weighted_isotonic_pools_only_monotonicity_violations() -> None:
    x, y = _weighted_isotonic(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.0, 1.0, 0.0, 1.0]),
    )

    assert x == (0.1, 0.2, 0.3, 0.4)
    assert y == (0.0, 0.5, 0.5, 1.0)
