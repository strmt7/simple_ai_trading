from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.fincast_runtime import fincast_context_valid_mask


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
