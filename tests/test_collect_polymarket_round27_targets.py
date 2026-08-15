from __future__ import annotations

import pytest

from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    Round27FeatureRow,
)
from tools.collect_polymarket_round27_targets import _role_rows


_START_MS = 1_786_784_400_000


def _row(index: int) -> Round27FeatureRow:
    event_start_ms = _START_MS + index * 300_000
    return Round27FeatureRow.create(
        run_id="stage1-a-run",
        condition_id="0x" + f"{index + 1:064x}",
        event_start_ms=event_start_ms,
        decision_time_ms=event_start_ms + 30_000,
        market_prior_probability=0.5,
        values=[0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES),
        maximum_receipt_wall_ms=event_start_ms + 29_999,
        source_chain_sha256="a" * 64,
    )


def test_target_operator_selects_only_the_exact_chronological_role() -> None:
    partitions = (
        {
            "role": "train",
            "slot_id": "stage1-a",
            "start_ms": _START_MS,
            "end_ms": _START_MS + 300_000,
        },
        {
            "role": "calibration",
            "slot_id": "stage1-a",
            "start_ms": _START_MS + 300_000,
            "end_ms": _START_MS + 600_000,
        },
    )

    selected = _role_rows(
        (_row(0), _row(1)),
        slot_id="stage1-a",
        role="calibration",
        partitions=partitions,
    )

    assert [row.condition_id for row in selected] == [_row(1).condition_id]

    with pytest.raises(ValueError, match="no feature rows"):
        _role_rows(
            (_row(0), _row(1)),
            slot_id="stage1-a",
            role="sealed",
            partitions=partitions,
        )
