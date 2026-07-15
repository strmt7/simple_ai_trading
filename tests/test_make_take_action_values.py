from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simple_ai_trading.make_take_action_features import (
    MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
    MakeTakeActionFeatureBatch,
    MakeTakeFeatureSpec,
)
from simple_ai_trading.make_take_action_values import build_make_take_action_values
from simple_ai_trading.make_take_payoff_lightgbm import MakeTakePayoffPredictionBatch
from simple_ai_trading.queue_fill_lightgbm import QueueFillPredictionBatch


def _sources():
    spec = MakeTakeFeatureSpec()
    actions = np.tile(np.arange(4, dtype=np.uint8), 2)
    sides = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), 2)
    eligible = np.asarray([True, False, True, True, True, True, True, True])
    features = MakeTakeActionFeatureBatch(
        schema_version=MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
        spec=spec,
        spec_sha256=spec.spec_sha256,
        source_dataset_sha256="1" * 64,
        source_flow_sha256="2" * 64,
        feature_names=("feature_a",),
        event_indexes=np.asarray([10, 11], dtype=np.int64),
        decision_time_ms=np.asarray([1_000, 2_000], dtype=np.int64),
        action_code=actions,
        action_side=sides,
        eligible=eligible,
        features=np.arange(8, dtype=np.float32).reshape(8, 1),
        batch_sha256="3" * 64,
    )
    passive_rows = np.asarray([0, 4, 5], dtype=np.int64)
    fill_probability = np.asarray([0.25, 0.50, 0.75])
    buckets = np.column_stack(
        (
            fill_probability,
            np.zeros(3),
            np.zeros(3),
            1.0 - fill_probability,
        )
    )
    fill = QueueFillPredictionBatch(
        source_action_feature_sha256=features.batch_sha256,
        source_panel_sha256="4" * 64,
        model_sha256="5" * 64,
        symbol="BTCUSDT",
        event_index=np.repeat(features.event_indexes, 4)[passive_rows],
        decision_time_ms=np.repeat(features.decision_time_ms, 4)[passive_rows],
        action_side=sides[passive_rows],
        hazard_probabilities=np.column_stack(
            (fill_probability, np.zeros(3), np.zeros(3))
        ),
        bucket_probabilities=buckets,
        fill_probability_15s=fill_probability,
    )
    mean = np.asarray([8.0, 6.0, 4.0, 2.0, -2.0, 10.0, 12.0, 14.0])
    payoff = MakeTakePayoffPredictionBatch(
        source_action_feature_sha256=features.batch_sha256,
        model_sha256="6" * 64,
        symbol="BTCUSDT",
        event_index=np.repeat(features.event_indexes, 4),
        decision_time_ms=np.repeat(features.decision_time_ms, 4),
        action_code=actions,
        action_side=sides,
        conditional_mean_bps=mean,
        conditional_q20_bps=mean - 5.0,
    )
    return features, fill, payoff


def test_action_values_use_fill_probability_only_for_passive_actions() -> None:
    features, fill, payoff = _sources()

    values = build_make_take_action_values(
        symbol="BTCUSDT",
        action_features=features,
        fill_predictions=fill,
        payoff_predictions=payoff,
    )

    np.testing.assert_allclose(
        values.fill_probability_15s,
        [0.25, 0.0, 1.0, 1.0, 0.50, 0.75, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        values.expected_mean_bps,
        [2.0, 0.0, 4.0, 2.0, -1.0, 7.5, 12.0, 14.0],
    )
    np.testing.assert_array_equal(
        values.conditional_q20_bps,
        payoff.conditional_q20_bps,
    )
    assert values.eligible[1] is np.False_
    with pytest.raises(ValueError, match="read-only"):
        values.expected_mean_bps[0] = 0.0


def test_action_values_reject_cross_batch_fill_predictions() -> None:
    features, fill, payoff = _sources()
    fill = replace(fill, source_action_feature_sha256="7" * 64)

    with pytest.raises(ValueError, match="source contract"):
        build_make_take_action_values(
            symbol="BTCUSDT",
            action_features=features,
            fill_predictions=fill,
            payoff_predictions=payoff,
        )
