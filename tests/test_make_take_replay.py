from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np

import simple_ai_trading.make_take_action_values as value_module
import simple_ai_trading.make_take_targets as target_module
from simple_ai_trading.make_take_action_values import (
    MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION,
    MakeTakeActionValueBatch,
)
from simple_ai_trading.make_take_payoff_panel import MAKE_TAKE_PAYOFF_SYMBOLS
from simple_ai_trading.make_take_replay import (
    build_make_take_fixed_ledger,
    replay_make_take_fixed_ledger,
    validate_make_take_fixed_ledger,
)
from simple_ai_trading.make_take_targets import (
    MAKE_TAKE_TARGET_SCHEMA_VERSION,
    MAKE_TAKE_UNFILLED_OUTCOME,
    MakeTakeTargetBatch,
)


def _action_values(symbol: str, source_sha: str) -> MakeTakeActionValueBatch:
    events = 3
    actions = np.tile(np.arange(4, dtype=np.uint8), events)
    sides = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), events)
    event_index = np.repeat(np.arange(events, dtype=np.int64), 4)
    decision_time = np.repeat(np.asarray([1_000, 3_000, 7_000]), 4)
    fill_probability = np.tile(np.asarray([0.5, 0.5, 1.0, 1.0]), events)
    conditional_mean = np.tile(np.asarray([20.0, 2.0, 5.0, 4.0]), events)
    conditional_q20 = np.tile(np.asarray([1.0, 1.0, 1.0, 1.0]), events)
    provisional = MakeTakeActionValueBatch(
        schema_version=MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION,
        symbol=symbol,
        source_dataset_sha256=source_sha,
        source_action_feature_sha256="1" * 64,
        source_fill_panel_sha256="2" * 64,
        fill_model_sha256="3" * 64,
        payoff_model_sha256="4" * 64,
        event_index=event_index,
        decision_time_ms=decision_time,
        action_code=actions,
        action_side=sides,
        eligible=np.ones(events * 4, dtype=np.bool_),
        fill_probability_15s=fill_probability,
        conditional_mean_bps=conditional_mean,
        conditional_q20_bps=conditional_q20,
        expected_mean_bps=fill_probability * conditional_mean,
        batch_sha256="",
    )
    return replace(
        provisional,
        batch_sha256=value_module._sha256(value_module._batch_payload(provisional)),
    )


def _targets(symbol: str, source_sha: str, *, scenario: str) -> MakeTakeTargetBatch:
    events = 3
    rows = events * 4
    actions = np.tile(np.arange(4, dtype=np.uint8), events)
    sides = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), events)
    filled = np.ones(rows, dtype=np.bool_)
    if scenario == "stress":
        filled[8] = False
    passive = actions < 2
    conditional = filled.copy()
    unfilled = passive & ~filled
    realized = conditional | unfilled
    conditional_net = np.ones(rows, dtype=np.float64)
    realized_net = np.ones(rows, dtype=np.float64)
    if scenario == "base":
        conditional_net[[0, 8]] = [5.0, 6.0]
        realized_net[[0, 8]] = [5.0, 6.0]
    else:
        conditional_net[0] = 4.0
        realized_net[0] = 4.0
        conditional_net[8] = np.nan
        realized_net[8] = 0.0
    terminal = np.repeat(np.asarray([5_000, 7_000, 9_000]), 4)
    if scenario == "stress":
        terminal[:4] = 6_000
        terminal[8] = 22_750
    outcome = np.zeros(rows, dtype=np.int8)
    outcome[unfilled] = MAKE_TAKE_UNFILLED_OUTCOME
    markout_5s = np.where(conditional, 1.0, np.nan)
    markout_15s = np.where(conditional, 2.0, np.nan)
    provisional = MakeTakeTargetBatch(
        schema_version=MAKE_TAKE_TARGET_SCHEMA_VERSION,
        scenario=scenario,
        symbol=symbol,
        source_dataset_sha256=source_sha,
        source_entry_sha256=("5" if scenario == "base" else "6") * 64,
        day_path_sha256=MappingProxyType({"0": "7" * 64}),
        event_rows=events,
        action_code=actions,
        action_side=sides,
        eligible=np.ones(rows, dtype=np.bool_),
        filled=filled,
        fill_bucket=np.where(filled & passive, 1, 0).astype(np.uint8),
        conditional_payoff_valid=conditional,
        realized_valid=realized,
        conditional_net_bps=conditional_net,
        realized_net_bps=realized_net,
        terminal_time_ms=terminal,
        outcome=outcome,
        markout_5s_bps=markout_5s,
        markout_15s_bps=markout_15s,
        stop_bps=np.full(rows, 40.0),
        take_bps=np.full(rows, 60.0),
        target_sha256="",
    )
    return replace(
        provisional,
        target_sha256=target_module._sha256(target_module._target_payload(provisional)),
    )


def _sources():
    source = {
        symbol: f"{index + 40:064x}"
        for index, symbol in enumerate(MAKE_TAKE_PAYOFF_SYMBOLS)
    }
    values = tuple(_action_values(symbol, source[symbol]) for symbol in source)
    base = tuple(_targets(symbol, source[symbol], scenario="base") for symbol in source)
    stress = tuple(
        _targets(symbol, source[symbol], scenario="stress") for symbol in source
    )
    return values, base, stress


def test_fixed_ledger_is_non_overlapping_and_reprices_identical_orders() -> None:
    values, base, stress = _sources()
    ledger = build_make_take_fixed_ledger(
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_mean_threshold_bps=1.0,
        conditional_q20_floor_bps=0.0,
    )
    base_metrics = replay_make_take_fixed_ledger(
        ledger,
        scenario="base",
        expected_days=[0],
    )
    stress_metrics = replay_make_take_fixed_ledger(
        ledger,
        scenario="stress",
        expected_days=[0],
    )

    assert ledger.selected_orders == 6
    assert {order.source_event_index for order in ledger.orders} == {0, 2}
    assert all(order.action_code == 0 for order in ledger.orders)
    assert base_metrics.closed_trades == 6
    assert base_metrics.total_net_bps == 33.0
    assert stress_metrics.closed_trades == 3
    assert stress_metrics.unfilled_orders == 3
    assert stress_metrics.total_net_bps == 12.0
    assert stress_metrics.maximum_single_symbol_positive_pnl_share == 1.0 / 3.0
    validate_make_take_fixed_ledger(ledger)


def test_fixed_ledger_can_abstain_without_fabricating_trades() -> None:
    values, base, stress = _sources()
    ledger = build_make_take_fixed_ledger(
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_mean_threshold_bps=11.0,
    )
    metrics = replay_make_take_fixed_ledger(
        ledger,
        scenario="base",
        expected_days=[0],
    )

    assert ledger.selected_orders == 0
    assert metrics.closed_trades == 0
    assert metrics.total_net_bps == 0.0
