from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.make_take_action_features import (
    MAKE_TAKE_ACTION_NAMES,
    build_make_take_action_features,
)
from simple_ai_trading.microstructure_features import AGGREGATE_DEPTH_FEATURE_NAMES
from simple_ai_trading.queue_censored_actions import build_exponential_flow_features


def _batch(*, decision_times: np.ndarray | None = None):
    decisions = (
        np.asarray([2_000, 3_000], dtype=np.int64)
        if decision_times is None
        else decision_times
    )
    source = np.zeros((2, len(AGGREGATE_DEPTH_FEATURE_NAMES)), dtype=np.float32)
    source[:, AGGREGATE_DEPTH_FEATURE_NAMES.index("return_60s_bps")] = [2.0, 3.0]
    source[:, AGGREGATE_DEPTH_FEATURE_NAMES.index("log_bid_l1_depth_quote")] = 7.0
    source[:, AGGREGATE_DEPTH_FEATURE_NAMES.index("log_ask_l1_depth_quote")] = 8.0
    source[
        :, AGGREGATE_DEPTH_FEATURE_NAMES.index("log_bid_notional_within_1pct")
    ] = 9.0
    source[
        :, AGGREGATE_DEPTH_FEATURE_NAMES.index("log_ask_notional_within_1pct")
    ] = 10.0
    source[
        :,
        AGGREGATE_DEPTH_FEATURE_NAMES.index(
            "aggregate_depth_notional_imbalance_1pct"
        ),
    ] = 0.25
    flow = build_exponential_flow_features(
        decision_time_ms=[2_000, 3_000],
        trade_time_ms=[500, 1_500],
        trade_price=[100.0, 100.0],
        trade_quantity=[1.0, 2.0],
        trade_buyer_is_maker=[False, True],
    )
    return build_make_take_action_features(
        source_features=source,
        source_feature_names=AGGREGATE_DEPTH_FEATURE_NAMES,
        decision_time_ms=decisions,
        bid_price=[100.0, 100.0],
        ask_price=[101.0, 101.0],
        bid_quantity=[100.0, 100.0],
        ask_quantity=[100.0, 100.0],
        flow=flow,
        source_dataset_sha256="a" * 64,
    )


def test_make_take_panel_is_event_major_directional_and_cost_aware() -> None:
    batch = _batch()

    assert batch.action_rows == 8
    assert batch.action_code.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert batch.action_side.tolist() == [1, -1, 1, -1, 1, -1, 1, -1]
    assert batch.summary()["eligible_by_action"] == {
        action: 2 for action in MAKE_TAKE_ACTION_NAMES
    }
    return_index = batch.feature_names.index("action_aligned_return_60s_bps")
    supporting_index = batch.feature_names.index("log_supporting_notional_within_1pct")
    opposing_index = batch.feature_names.index("log_opposing_notional_within_1pct")
    imbalance_index = batch.feature_names.index(
        "action_aligned_aggregate_depth_notional_imbalance_1pct"
    )
    flow_index = batch.feature_names.index("action_aligned_flow_imbalance_h1s")
    passive_index = batch.feature_names.index("action_is_passive")
    queue_index = batch.feature_names.index("log_queue_ahead_quote")
    cost_index = batch.feature_names.index("known_round_trip_cost_bps")
    assert batch.features[:4, return_index].tolist() == [2.0, -2.0, 2.0, -2.0]
    assert batch.features[0, supporting_index] == 9.0
    assert batch.features[1, supporting_index] == 10.0
    assert batch.features[1, opposing_index] == 9.0
    assert batch.features[0, imbalance_index] == 0.25
    assert batch.features[1, imbalance_index] == -0.25
    assert batch.features[0, flow_index] == -batch.features[1, flow_index]
    assert batch.features[:4, passive_index].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert np.all(batch.features[:2, queue_index] > 0.0)
    assert np.all(batch.features[2:4, queue_index] == 0.0)
    assert np.all(batch.features[2:4, cost_index] > batch.features[:2, cost_index])
    assert batch.features[0, cost_index] == pytest.approx(9.0)
    assert batch.features[2, cost_index] == pytest.approx(
        (1.0 - 100.0 / 101.0) * 10_000.0 + 6.0 + 6.0 * (100.0 / 101.0)
    )
    assert batch.features[3, cost_index] == pytest.approx(
        (101.0 / 100.0 - 1.0) * 10_000.0 + 6.0 + 6.0 * (101.0 / 100.0)
    )


def test_make_take_panel_is_hash_bound_and_immutable() -> None:
    first = _batch()
    second = _batch()

    assert first.batch_sha256 == second.batch_sha256
    assert len(first.spec_sha256) == 64
    assert len(first.source_flow_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        first.features[0, 0] = 1.0


def test_make_take_panel_rejects_flow_time_or_feature_contract_drift() -> None:
    with pytest.raises(ValueError, match="decision times"):
        _batch(decision_times=np.asarray([2_000, 4_000], dtype=np.int64))
