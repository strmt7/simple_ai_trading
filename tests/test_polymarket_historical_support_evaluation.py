from __future__ import annotations

import numpy as np

from simple_ai_trading.polymarket_historical_dataset import FEATURE_NAMES
from simple_ai_trading.polymarket_historical_model import HistoricalModelPanel
from simple_ai_trading.polymarket_historical_support import (
    HistoricalFeatureSupportProfile,
)
from simple_ai_trading.polymarket_historical_support_evaluation import (
    evaluate_historical_support_role,
)


def _panel() -> HistoricalModelPanel:
    condition_ids = np.repeat(
        np.asarray(
            [f"condition-{index}" for index in range(8)],
            dtype=object,
        ),
        8,
    )
    labels = np.repeat(np.asarray([0.0, 1.0] * 4), 8)
    rows = len(labels)
    features = np.zeros((rows, len(FEATURE_NAMES)), dtype=np.float32)
    features[-1, 0] = 100.0
    panel = HistoricalModelPanel(
        condition_ids=condition_ids,
        roles=np.full(rows, "test", dtype=object),
        event_start_ms=np.repeat(
            np.arange(8, dtype=np.int64) * 300_000 + 1_000_000,
            8,
        ),
        decision_time_ms=(
            np.repeat(
                np.arange(8, dtype=np.int64) * 300_000 + 1_000_000,
                8,
            )
            + np.tile(
                np.arange(30_000, 270_000, 30_000, dtype=np.int64),
                8,
            )
        ),
        features=features,
        labels=labels,
        dataset_sha256="1" * 64,
    )
    panel.validate(expected_roles=("test",))
    return panel


def _profile() -> HistoricalFeatureSupportProfile:
    width = len(FEATURE_NAMES)
    return HistoricalFeatureSupportProfile(
        artifact_sha256="2" * 64,
        pretest_artifact_sha256="3" * 64,
        dataset_sha256="1" * 64,
        source_commit="4" * 40,
        feature_names=FEATURE_NAMES,
        training_rows=64,
        training_conditions=8,
        minimum=np.full(width, -1.0),
        maximum=np.full(width, 1.0),
        outer_lower=np.full(width, -10.0),
        outer_upper=np.full(width, 10.0),
        maximum_outside_training_range=4,
        maximum_extreme_outliers=0,
    )


def test_role_evaluation_records_coverage_without_improvement_claim() -> None:
    panel = _panel()
    probability = np.where(panel.labels == 1.0, 0.8, 0.2)
    report = evaluate_historical_support_role(
        panel,
        probability,
        _profile(),
    )
    assert report["rows"] == 64
    assert report["admitted_rows"] == 63
    assert report["abstained_rows"] == 1
    assert report["conditions_with_any_admitted_row"] == 8
    assert report["conditions_with_all_rows_admitted"] == 7
    assert report["abstention_causes"] == {
        "outside_training_range_limit": 0,
        "extreme_outer_fence": 1,
        "both": 0,
    }
    assert report["decision_offset_seconds"]["240"]["abstained"] == 1
    assert report["extreme_outer_fence_feature_counts"] == {
        FEATURE_NAMES[0]: 1
    }
