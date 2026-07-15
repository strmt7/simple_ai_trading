from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simple_ai_trading.make_take_action_features import (
    build_make_take_action_features,
)
from simple_ai_trading.make_take_scenario_entries import (
    build_make_take_scenario_entries,
)
from simple_ai_trading.microstructure_features import AGGREGATE_DEPTH_FEATURE_NAMES
from simple_ai_trading.queue_censored_actions import (
    build_exponential_flow_features,
    build_passive_fill_result,
)
from simple_ai_trading.queue_fill_survival import (
    build_hazard_risk_set,
    build_passive_fill_survival_panel,
    evaluate_fill_survival_probabilities,
    fill_bucket_prevalence,
    hazards_to_bucket_probabilities,
)


def _panel():
    decisions = np.asarray([10_000, 30_000], dtype=np.int64)
    arrivals = decisions + 750
    trade_time = [12_000, 37_000]
    trade_price = [100.0, 100.1]
    trade_quantity = [120.0, 120.0]
    trade_side = [True, False]
    common = {
        "arrival_time_ms": arrivals,
        "queue_ahead_quantity": [100.0, 100.0],
        "order_notional_quote": 1_000.0,
        "trade_id": [1, 2],
        "trade_time_ms": trade_time,
        "trade_price": trade_price,
        "trade_quantity": trade_quantity,
        "trade_buyer_is_maker": trade_side,
    }
    long_fill = build_passive_fill_result(
        placement_price=[100.0, 100.0], buyer_is_maker=True, **common
    )
    short_fill = build_passive_fill_result(
        placement_price=[100.1, 100.1], buyer_is_maker=False, **common
    )
    entries = build_make_take_scenario_entries(
        scenario="base",
        decision_time_ms=decisions,
        bid_price=[100.0, 100.0],
        ask_price=[100.1, 100.1],
        bid_quantity=[100.0, 100.0],
        ask_quantity=[100.0, 100.0],
        long_fill=long_fill,
        short_fill=short_fill,
    )
    flow = build_exponential_flow_features(
        decision_time_ms=decisions,
        trade_time_ms=trade_time,
        trade_price=trade_price,
        trade_quantity=trade_quantity,
        trade_buyer_is_maker=trade_side,
    )
    source = np.zeros((2, len(AGGREGATE_DEPTH_FEATURE_NAMES)), dtype=np.float32)
    features = build_make_take_action_features(
        source_features=source,
        source_feature_names=AGGREGATE_DEPTH_FEATURE_NAMES,
        decision_time_ms=decisions,
        bid_price=[100.0, 100.0],
        ask_price=[100.1, 100.1],
        bid_quantity=[100.0, 100.0],
        ask_quantity=[100.0, 100.0],
        flow=flow,
        source_dataset_sha256="a" * 64,
    )
    return build_passive_fill_survival_panel(features, entries)


def test_survival_panel_and_hazard_risk_sets_preserve_censoring() -> None:
    panel = _panel()

    assert panel.action_side.tolist() == [1, -1, 1, -1]
    assert panel.fill_bucket.tolist() == [1, 0, 0, 2]
    first = build_hazard_risk_set(panel, 0)
    second = build_hazard_risk_set(panel, 1)
    third = build_hazard_risk_set(panel, 2)
    assert first.labels.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert second.source_rows.tolist() == [1, 2, 3]
    assert second.labels.tolist() == [0.0, 0.0, 1.0]
    assert third.source_rows.tolist() == [1, 2]
    assert third.labels.tolist() == [0.0, 0.0]
    assert len(panel.panel_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        panel.fill_bucket[0] = 0


def test_hazards_form_normalized_bucket_distribution_and_proper_scores() -> None:
    panel = _panel()
    hazards = np.asarray(
        [
            [0.99, 0.50, 0.50],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
            [0.01, 0.99, 0.50],
        ]
    )
    probabilities = hazards_to_bucket_probabilities(hazards)
    metrics = evaluate_fill_survival_probabilities(
        panel,
        probabilities,
        fill_bucket_prevalence(panel),
    )

    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
    assert metrics["log_loss_skill"] > 0.0
    assert metrics["integrated_brier_skill"] > 0.0
    assert metrics["observed_fill_ratio"] == 0.5


def test_survival_panel_fails_closed_on_entry_alignment_drift() -> None:
    panel = _panel()
    with pytest.raises(ValueError, match="hazard index"):
        build_hazard_risk_set(panel, True)
    invalid = replace(panel, fill_bucket=np.asarray([4, 0, 0, 0], dtype=np.uint8))
    with pytest.raises(ValueError, match="probability matrix"):
        hazards_to_bucket_probabilities(np.ones((invalid.rows, 2)))
