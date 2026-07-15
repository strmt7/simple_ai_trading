from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import simple_ai_trading.make_take_action_values as value_module
import simple_ai_trading.make_take_policy as policy_module
import simple_ai_trading.make_take_targets as target_module
from simple_ai_trading.make_take_payoff_panel import MAKE_TAKE_PAYOFF_SYMBOLS
from simple_ai_trading.make_take_policy import (
    calibrate_make_take_policy,
    validate_make_take_policy_selection,
)
from simple_ai_trading.make_take_predictive_evaluation import (
    MAKE_TAKE_PREDICTIVE_EVALUATION_SCHEMA_VERSION,
    FillPredictiveMetric,
    MakeTakePredictiveEvaluation,
    PayoffPredictiveMetric,
)


_DAY_MS = 86_400_000


def _load_replay_helpers():
    path = Path(__file__).with_name("test_make_take_replay.py")
    specification = importlib.util.spec_from_file_location("make_take_replay_helpers", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def _two_day_sources():
    helper = _load_replay_helpers()
    raw_values, raw_base, raw_stress = helper._sources()
    values = []
    base = []
    stress = []
    for symbol_index, (value, base_target, stress_target) in enumerate(
        zip(raw_values, raw_base, raw_stress, strict=True)
    ):
        decision_time = value.decision_time_ms.copy()
        decision_time[8:12] = _DAY_MS + 1_000
        conditional_mean = value.conditional_mean_bps.copy()
        conditional_mean[0] = 4.0 + 2.0 * symbol_index
        conditional_mean[4:8] = [2.0, 2.0, 1.0, 1.0]
        conditional_mean[8] = 16.0 + 2.0 * symbol_index
        expected_mean = value.fill_probability_15s * conditional_mean
        provisional_value = replace(
            value,
            decision_time_ms=decision_time,
            conditional_mean_bps=conditional_mean,
            expected_mean_bps=expected_mean,
            batch_sha256="",
        )
        values.append(
            replace(
                provisional_value,
                batch_sha256=value_module._sha256(
                    value_module._batch_payload(provisional_value)
                ),
            )
        )

        base_terminal = base_target.terminal_time_ms.copy()
        base_terminal[8:12] += _DAY_MS
        base_conditional = base_target.conditional_net_bps.copy()
        base_realized = base_target.realized_net_bps.copy()
        base_conditional[[0, 8]] = [1.0 + symbol_index, 8.0 + symbol_index]
        base_realized[[0, 8]] = [1.0 + symbol_index, 8.0 + symbol_index]
        provisional_base = replace(
            base_target,
            terminal_time_ms=base_terminal,
            conditional_net_bps=base_conditional,
            realized_net_bps=base_realized,
            target_sha256="",
        )
        base.append(
            replace(
                provisional_base,
                target_sha256=target_module._sha256(
                    target_module._target_payload(provisional_base)
                ),
            )
        )

        stress_filled = stress_target.filled.copy()
        stress_filled[8] = True
        stress_bucket = stress_target.fill_bucket.copy()
        stress_bucket[8] = 1
        stress_conditional_valid = stress_target.conditional_payoff_valid.copy()
        stress_conditional_valid[8] = True
        stress_conditional = stress_target.conditional_net_bps.copy()
        stress_realized = stress_target.realized_net_bps.copy()
        stress_conditional[[0, 8]] = [
            0.5 + 0.5 * symbol_index,
            6.0 + symbol_index,
        ]
        stress_realized[[0, 8]] = [
            0.5 + 0.5 * symbol_index,
            6.0 + symbol_index,
        ]
        stress_terminal = stress_target.terminal_time_ms.copy()
        stress_terminal[8:12] = _DAY_MS + 6_000
        stress_outcome = stress_target.outcome.copy()
        stress_outcome[8] = 0
        markout_5s = stress_target.markout_5s_bps.copy()
        markout_15s = stress_target.markout_15s_bps.copy()
        markout_5s[8] = 1.0
        markout_15s[8] = 2.0
        provisional_stress = replace(
            stress_target,
            filled=stress_filled,
            fill_bucket=stress_bucket,
            conditional_payoff_valid=stress_conditional_valid,
            conditional_net_bps=stress_conditional,
            realized_net_bps=stress_realized,
            terminal_time_ms=stress_terminal,
            outcome=stress_outcome,
            markout_5s_bps=markout_5s,
            markout_15s_bps=markout_15s,
            target_sha256="",
        )
        stress.append(
            replace(
                provisional_stress,
                target_sha256=target_module._sha256(
                    target_module._target_payload(provisional_stress)
                ),
            )
        )
    return tuple(values), tuple(base), tuple(stress)


def _predictive_report() -> MakeTakePredictiveEvaluation:
    fill_metrics = tuple(
        FillPredictiveMetric(
            symbol=symbol,
            side=side,
            rows=100,
            log_loss=0.9,
            baseline_log_loss=1.0,
            log_loss_skill=0.1,
            integrated_brier=0.18,
            baseline_integrated_brier=0.20,
            integrated_brier_skill=0.1,
            predicted_fill_probability=0.5,
            observed_fill_ratio=0.5,
            absolute_calibration_error=0.0,
            passed=True,
        )
        for symbol in sorted(MAKE_TAKE_PAYOFF_SYMBOLS)
        for side in ("long", "short")
    )
    payoff_metrics = tuple(
        PayoffPredictiveMetric(
            symbol=symbol,
            action_code=action,
            action_name=(
                "passive_long",
                "passive_short",
                "aggressive_long",
                "aggressive_short",
            )[action],
            rows=100,
            mean_mse_bps2=1.0,
            baseline_mean_mse_bps2=2.0,
            mean_mse_skill=0.5,
            q20_pinball_bps=0.5,
            baseline_q20_pinball_bps=1.0,
            q20_pinball_skill=0.5,
            spearman=0.2,
            top_quintile_rows=20,
            top_quintile_mean_net_bps=2.0,
            top_quintile_mean_markout_5s_bps=1.0,
            top_quintile_mean_markout_15s_bps=1.0,
            passed=True,
        )
        for symbol in sorted(MAKE_TAKE_PAYOFF_SYMBOLS)
        for action in range(4)
    )
    panel_map = tuple(
        (symbol, f"{index + 50:064x}")
        for index, symbol in enumerate(sorted(MAKE_TAKE_PAYOFF_SYMBOLS))
    )
    provisional = MakeTakePredictiveEvaluation(
        schema_version=MAKE_TAKE_PREDICTIVE_EVALUATION_SCHEMA_VERSION,
        role="policy_calibration",
        fill_model_sha256="3" * 64,
        payoff_model_sha256="4" * 64,
        training_fill_panel_sha256_by_symbol=panel_map,
        evaluation_fill_panel_sha256_by_symbol=panel_map,
        training_payoff_panel_sha256_by_symbol=panel_map,
        evaluation_payoff_panel_sha256_by_symbol=panel_map,
        fill_metrics=fill_metrics,
        payoff_metrics=payoff_metrics,
        payoff_early_quality_gate_passed=True,
        predictive_gate_passed=True,
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=policy_module._sha256(
            policy_module.asdict(provisional) | {"report_sha256": ""}
        ),
    )


def _report_with_valid_hash() -> MakeTakePredictiveEvaluation:
    report = _predictive_report()
    payload = policy_module.asdict(report)
    payload.pop("report_sha256")
    return replace(report, report_sha256=policy_module._sha256(payload))


def test_policy_selects_only_threshold_that_keeps_both_days_positive() -> None:
    values, base, stress = _two_day_sources()
    selection = calibrate_make_take_policy(
        predictive_evaluation=_report_with_valid_hash(),
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_days=[0, 1],
    )

    assert selection.accepted is True
    assert selection.selected_coverage_quantile == 0.0
    assert selection.selected_ledger is not None
    assert selection.action_value_quintile.passed is True
    assert selection.candidates[0].accepted is True
    assert all(
        any(
            reason.endswith("constituent_day_not_positive")
            for reason in candidate.rejection_reasons
        )
        for candidate in selection.candidates[1:]
    )
    validate_make_take_policy_selection(selection)
