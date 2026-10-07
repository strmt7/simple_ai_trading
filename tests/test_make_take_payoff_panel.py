from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from simple_ai_trading.make_take_action_features import (
    MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
    MakeTakeActionFeatureBatch,
    MakeTakeFeatureSpec,
)
from simple_ai_trading.make_take_payoff_panel import (
    build_make_take_conditional_payoff_panel,
    validate_make_take_conditional_payoff_panel,
)
from simple_ai_trading.make_take_scenario_entries import (
    build_make_take_scenario_entries,
)
from simple_ai_trading.make_take_targets import (
    MAKE_TAKE_TARGET_SCHEMA_VERSION,
    MakeTakeTargetBatch,
    _target_payload,
    _sha256,
)
from simple_ai_trading.queue_censored_actions import build_passive_fill_result


def _sources():
    action_code = np.tile(np.arange(4, dtype=np.uint8), 2)
    action_side = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), 2)
    decisions = np.asarray([1_000, 11_000], dtype=np.int64)
    eligible = np.ones(8, dtype=np.bool_)
    spec = MakeTakeFeatureSpec()
    features = MakeTakeActionFeatureBatch(
        schema_version=MAKE_TAKE_ACTION_FEATURE_SCHEMA_VERSION,
        spec=spec,
        spec_sha256=spec.spec_sha256,
        source_dataset_sha256="a" * 64,
        source_flow_sha256="b" * 64,
        feature_names=("feature_a", "feature_b"),
        event_indexes=np.asarray([0, 1], dtype=np.int64),
        decision_time_ms=decisions,
        action_code=action_code,
        action_side=action_side,
        eligible=eligible,
        features=np.arange(16, dtype=np.float32).reshape(8, 2),
        batch_sha256="c" * 64,
    )
    common = {
        "arrival_time_ms": decisions + 750,
        "queue_ahead_quantity": [100.0, 100.0],
        "order_notional_quote": 1000.0,
        "trade_id": [1, 2],
        "trade_time_ms": [2000, 17000],
        "trade_price": [100.0, 100.1],
        "trade_quantity": [120.0, 120.0],
        "trade_buyer_is_maker": [True, False],
    }
    entries = build_make_take_scenario_entries(
        scenario="base",
        decision_time_ms=decisions,
        bid_price=[100.0, 100.0],
        ask_price=[100.1, 100.1],
        bid_quantity=[100.0, 100.0],
        ask_quantity=[100.0, 100.0],
        long_fill=build_passive_fill_result(
            placement_price=[100.0, 100.0], buyer_is_maker=True, **common
        ),
        short_fill=build_passive_fill_result(
            placement_price=[100.1, 100.1], buyer_is_maker=False, **common
        ),
    )
    filled, fill_bucket = entries.filled, entries.fill_bucket
    conditional_valid = filled.copy()
    net = np.where(conditional_valid, np.arange(8, dtype=np.float64) - 4.0, np.nan)
    markout_5s = np.where(conditional_valid, 2.0, np.nan)
    markout_15s = np.where(conditional_valid, 3.0, np.nan)
    terminal = np.where(
        conditional_valid,
        np.repeat(decisions + 300_750, 4),
        entries.unfilled_expiry_time_ms,
    )
    targets = MakeTakeTargetBatch(
        schema_version=MAKE_TAKE_TARGET_SCHEMA_VERSION,
        scenario="base",
        symbol="BTCUSDT",
        source_dataset_sha256="a" * 64,
        source_entry_sha256=entries.batch_sha256,
        day_path_sha256=MappingProxyType({"0": "1" * 64}),
        event_rows=2,
        action_code=action_code,
        action_side=action_side,
        eligible=eligible,
        filled=filled,
        fill_bucket=fill_bucket,
        conditional_payoff_valid=conditional_valid,
        realized_valid=np.ones(8, dtype=np.bool_),
        conditional_net_bps=net,
        realized_net_bps=np.nan_to_num(net),
        terminal_time_ms=terminal,
        outcome=np.where(conditional_valid, 0, -2).astype(np.int8),
        markout_5s_bps=markout_5s,
        markout_15s_bps=markout_15s,
        stop_bps=np.full(8, 40.0),
        take_bps=np.full(8, 60.0),
        target_sha256="2" * 64,
    )
    targets = replace(targets, target_sha256=_sha256(_target_payload(targets)))
    return features, entries, targets


def test_payoff_panel_excludes_unfilled_passive_orders_and_binds_horizon() -> None:
    features, entries, targets = _sources()

    panel = build_make_take_conditional_payoff_panel(
        symbol="BTCUSDT",
        action_features=features,
        entries=entries,
        targets=targets,
    )

    assert panel.rows == 6
    assert panel.action_code.tolist() == [0, 2, 3, 1, 2, 3]
    assert panel.event_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert panel.source_label_end_ms == 326_750
    assert panel.summary()["rows_by_action"] == {
        "passive_long": 1,
        "passive_short": 1,
        "aggressive_long": 2,
        "aggressive_short": 2,
    }
    validate_make_take_conditional_payoff_panel(panel)
    with pytest.raises(ValueError, match="read-only"):
        panel.net_bps[0] = 0.0


def test_payoff_panel_rejects_feature_entry_alignment_drift() -> None:
    features, entries, targets = _sources()
    drifted_side = entries.action_side.copy()
    drifted_side[0] = -1
    entries = replace(entries, action_side=drifted_side)

    with pytest.raises(ValueError, match="entry identity or lifecycle"):
        build_make_take_conditional_payoff_panel(
            symbol="BTCUSDT",
            action_features=features,
            entries=entries,
            targets=targets,
        )
