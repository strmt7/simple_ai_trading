from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import MappingProxyType

import numpy as np

import simple_ai_trading.make_take_action_values as value_module
import simple_ai_trading.make_take_predictive_evaluation as predictive_module
import simple_ai_trading.make_take_targets as target_module
from simple_ai_trading.make_take_action_values import (
    MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION,
    MakeTakeActionValueBatch,
)
from simple_ai_trading.make_take_evaluation import (
    evaluate_make_take_policy,
    validate_make_take_economic_evaluation,
)
from simple_ai_trading.make_take_payoff_panel import MAKE_TAKE_PAYOFF_SYMBOLS
from simple_ai_trading.make_take_targets import (
    MAKE_TAKE_TARGET_SCHEMA_VERSION,
    MakeTakeTargetBatch,
)


_DAY_MS = 86_400_000


def _load_policy_helpers():
    path = Path(__file__).with_name("test_make_take_policy.py")
    specification = importlib.util.spec_from_file_location("make_take_policy_helpers", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def _accepted_policy():
    helper = _load_policy_helpers()
    values, base, stress = helper._two_day_sources()
    selection = helper.calibrate_make_take_policy(
        predictive_evaluation=helper._report_with_valid_hash(),
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_days=[0, 1],
    )
    return helper, selection, values, base, stress


def _evaluation_report(helper):
    source = helper._report_with_valid_hash()
    provisional = replace(source, role="evaluation", report_sha256="")
    return replace(
        provisional,
        report_sha256=predictive_module._sha256(
            predictive_module._report_payload(provisional)
        ),
    )


def _evaluation_values(symbol: str, source_sha: str) -> MakeTakeActionValueBatch:
    events = 11
    event_times = np.concatenate(
        (
            1_000 + np.arange(6, dtype=np.int64) * 400_000,
            _DAY_MS + 1_000 + np.arange(5, dtype=np.int64) * 400_000,
        )
    )
    actions = np.tile(np.arange(4, dtype=np.uint8), events)
    sides = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), events)
    fill_probability = np.tile(np.asarray([0.5, 0.5, 1.0, 1.0]), events)
    conditional_mean = np.tile(np.asarray([20.0, 2.0, 5.0, 4.0]), events)
    conditional_q20 = np.ones(events * 4, dtype=np.float64)
    provisional = MakeTakeActionValueBatch(
        schema_version=MAKE_TAKE_ACTION_VALUE_SCHEMA_VERSION,
        symbol=symbol,
        source_dataset_sha256=source_sha,
        source_action_feature_sha256="1" * 64,
        source_fill_panel_sha256="2" * 64,
        fill_model_sha256="3" * 64,
        payoff_model_sha256="4" * 64,
        event_index=np.repeat(np.arange(events, dtype=np.int64), 4),
        decision_time_ms=np.repeat(event_times, 4),
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


def _evaluation_targets(
    symbol: str,
    source_sha: str,
    *,
    symbol_index: int,
    scenario: str,
) -> MakeTakeTargetBatch:
    events = 11
    rows = events * 4
    event_times = np.concatenate(
        (
            1_000 + np.arange(6, dtype=np.int64) * 400_000,
            _DAY_MS + 1_000 + np.arange(5, dtype=np.int64) * 400_000,
        )
    )
    actions = np.tile(np.arange(4, dtype=np.uint8), events)
    sides = np.tile(np.asarray([1, -1, 1, -1], dtype=np.int8), events)
    net = np.ones(rows, dtype=np.float64)
    selected_net = (2.0 if scenario == "base" else 1.0) + symbol_index
    net[0::4] = selected_net
    provisional = MakeTakeTargetBatch(
        schema_version=MAKE_TAKE_TARGET_SCHEMA_VERSION,
        scenario=scenario,
        symbol=symbol,
        source_dataset_sha256=source_sha,
        source_entry_sha256=("5" if scenario == "base" else "6") * 64,
        day_path_sha256=MappingProxyType(
            {"0": "7" * 64, str(_DAY_MS): "8" * 64}
        ),
        event_rows=events,
        action_code=actions,
        action_side=sides,
        eligible=np.ones(rows, dtype=np.bool_),
        filled=np.ones(rows, dtype=np.bool_),
        fill_bucket=np.where(actions < 2, 1, 0).astype(np.uint8),
        conditional_payoff_valid=np.ones(rows, dtype=np.bool_),
        realized_valid=np.ones(rows, dtype=np.bool_),
        conditional_net_bps=net,
        realized_net_bps=net.copy(),
        terminal_time_ms=np.repeat(event_times + 300_000, 4),
        outcome=np.zeros(rows, dtype=np.int8),
        markout_5s_bps=np.ones(rows, dtype=np.float64),
        markout_15s_bps=np.ones(rows, dtype=np.float64),
        stop_bps=np.full(rows, 40.0),
        take_bps=np.full(rows, 60.0),
        target_sha256="",
    )
    return replace(
        provisional,
        target_sha256=target_module._sha256(target_module._target_payload(provisional)),
    )


def _terminal_sources():
    source = {
        symbol: f"{index + 60:064x}"
        for index, symbol in enumerate(MAKE_TAKE_PAYOFF_SYMBOLS)
    }
    values = tuple(
        _evaluation_values(symbol, source[symbol])
        for symbol in MAKE_TAKE_PAYOFF_SYMBOLS
    )
    base = tuple(
        _evaluation_targets(
            symbol,
            source[symbol],
            symbol_index=index,
            scenario="base",
        )
        for index, symbol in enumerate(MAKE_TAKE_PAYOFF_SYMBOLS)
    )
    stress = tuple(
        _evaluation_targets(
            symbol,
            source[symbol],
            symbol_index=index,
            scenario="stress",
        )
        for index, symbol in enumerate(MAKE_TAKE_PAYOFF_SYMBOLS)
    )
    return values, base, stress


def test_terminal_evaluation_rejects_insufficient_closed_trade_support() -> None:
    helper, selection, values, base, stress = _accepted_policy()
    evaluation = evaluate_make_take_policy(
        policy_selection=selection,
        predictive_evaluation=_evaluation_report(helper),
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_days=range(6),
    )

    assert evaluation.economic_gate_passed is False
    assert "base_minimum_closed_trades_not_met" in evaluation.rejection_reasons
    assert "stress_minimum_closed_trades_not_met" in evaluation.rejection_reasons
    assert (
        evaluation.evaluation_ledger.expected_mean_threshold_bps
        == selection.selected_expected_mean_threshold_bps
    )


def test_terminal_evaluation_passes_unchanged_policy_with_33_stress_trades() -> None:
    helper, selection, _values, _base, _stress = _accepted_policy()
    values, base, stress = _terminal_sources()
    evaluation = evaluate_make_take_policy(
        policy_selection=selection,
        predictive_evaluation=_evaluation_report(helper),
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_days=range(6),
    )

    assert evaluation.economic_gate_passed is True
    assert evaluation.base_metrics.closed_trades == 33
    assert evaluation.stress_metrics.closed_trades == 33
    assert evaluation.base_metrics.maximum_drawdown_bps == 0.0
    assert evaluation.stress_metrics.maximum_drawdown_bps == 0.0
    assert evaluation.rejection_reasons == ()
    validate_make_take_economic_evaluation(evaluation)
