from __future__ import annotations

import pytest

from simple_ai_trading.microstructure_availability import SnapshotAvailability


@pytest.mark.parametrize(
    ("receipt", "offset", "boundary"),
    [
        (1_700_000_000_000_000_000, 0.0, 1_700_000_000_001),
        (1_700_000_000_000_000_001, 0.0, 1_700_000_000_001),
        (1_700_000_000_000_000_001, 0.25, 1_700_000_000_001),
        (1_700_000_000_000_000_001, -0.25, 1_700_000_000_000),
        (1_700_000_000_000_750_000, -0.25, 1_700_000_000_001),
        (1_700_000_000_000_500_000, 0.5, 1_700_000_000_002),
    ],
)
def test_snapshot_boundary_preserves_nanoseconds_and_rounds_up(
    receipt: int, offset: float, boundary: int
) -> None:
    assert SnapshotAvailability(receipt, offset).decision_not_before_ms == boundary


@pytest.mark.parametrize("receipt", [True, False, 0, -1, 1.5, "100", None])
def test_snapshot_boundary_rejects_missing_or_lossy_receipts(receipt: object) -> None:
    with pytest.raises(ValueError, match="positive native integer"):
        SnapshotAvailability(receipt, 0.0).decision_not_before_ms


@pytest.mark.parametrize(
    "offset", [True, "0", float("nan"), float("inf"), -60_001, 60_001]
)
def test_snapshot_boundary_rejects_invalid_offsets(offset: object) -> None:
    with pytest.raises(ValueError, match="offset"):
        SnapshotAvailability(1_700_000_000_000_000_000, offset).decision_not_before_ms


def test_snapshot_boundary_rejects_nonpositive_estimated_exchange_time() -> None:
    with pytest.raises(ValueError, match="boundary must be positive"):
        SnapshotAvailability(1, -1.0).decision_not_before_ms
