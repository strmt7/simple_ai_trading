from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_architecture import (
    ActionValueEnsembleBatch,
)
from simple_ai_trading.microstructure_three_action_lightgbm import (
    ThreeActionEnsembleBatch,
    as_selective_action_ensemble,
)
from tools.run_three_action_viability import (
    _architecture_gate_reasons,
    _calibration_architecture_diagnostics,
    load_round34_design,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-034-three-action-utility-design.json"
)


def _diagnostic_bundle() -> tuple[SimpleNamespace, ThreeActionEnsembleBatch]:
    rows = 20
    endpoints = np.arange(rows, dtype=np.int64)
    opportunity = endpoints < 10
    long_preferred = endpoints % 2 == 0
    magnitude = np.arange(4.0, 24.0)
    long_actual = np.where(
        opportunity,
        np.where(long_preferred, magnitude, -magnitude),
        -0.1,
    )
    short_actual = np.where(
        opportunity,
        np.where(long_preferred, -magnitude, magnitude),
        -0.1,
    )
    long_prediction = np.where(long_preferred, magnitude * 0.8, -magnitude * 0.8)
    short_prediction = -long_prediction
    long_profitable = np.where(long_actual > 0.0, 0.9, 0.1)
    short_profitable = np.where(short_actual > 0.0, 0.9, 0.1)
    long_action = np.where(
        opportunity & long_preferred,
        0.85,
        0.05,
    )
    short_action = np.where(
        opportunity & ~long_preferred,
        0.85,
        0.05,
    )
    abstain_action = 1.0 - long_action - short_action
    opportunity_probability = long_action + short_action
    conditional_long = long_action / opportunity_probability
    standard_deviation = np.full(rows, 0.1, dtype=np.float64)
    action = ActionValueEnsembleBatch(
        endpoint_indexes=endpoints,
        long_mean_bps=long_prediction,
        short_mean_bps=short_prediction,
        long_epistemic_std_bps=standard_deviation,
        short_epistemic_std_bps=standard_deviation,
        long_profitable_probability=long_profitable,
        short_profitable_probability=short_profitable,
        long_lower_bps=long_prediction - 1.0,
        short_lower_bps=short_prediction - 1.0,
        long_upper_bps=long_prediction + 1.0,
        short_upper_bps=short_prediction + 1.0,
        long_positive_member_ratio=np.where(long_preferred, 1.0, 0.0),
        short_positive_member_ratio=np.where(long_preferred, 0.0, 1.0),
        member_count=3,
    )
    long_members = np.repeat(long_action[None, :], 3, axis=0)
    abstain_members = np.repeat(abstain_action[None, :], 3, axis=0)
    short_members = np.repeat(short_action[None, :], 3, axis=0)
    opportunity_members = long_members + short_members
    direction_members = long_members / opportunity_members
    ensemble = ThreeActionEnsembleBatch(
        action_values=action,
        long_action_probability_mean=long_action,
        abstain_action_probability_mean=abstain_action,
        short_action_probability_mean=short_action,
        opportunity_probability_mean=opportunity_probability,
        opportunity_probability_std=np.zeros(rows),
        conditional_long_probability_mean=conditional_long,
        conditional_long_probability_std=np.zeros(rows),
        long_action_member_probabilities=long_members,
        abstain_action_member_probabilities=abstain_members,
        short_action_member_probabilities=short_members,
        opportunity_member_probabilities=opportunity_members,
        conditional_long_member_probabilities=direction_members,
        direction_long_member_ratio=np.mean(direction_members > 0.5, axis=0),
        direction_short_member_ratio=np.mean(direction_members < 0.5, axis=0),
        side_consensus_member_ratio=np.ones(rows),
        member_count=3,
    )
    targets = SimpleNamespace(
        source_indexes=endpoints,
        rows=rows,
        valid=np.ones(rows, dtype=bool),
        stress_long_net_bps=long_actual,
        stress_short_net_bps=short_actual,
    )
    return targets, ensemble


def test_round34_runner_loads_exact_frozen_design() -> None:
    design, design_sha, profiles = load_round34_design(DESIGN)

    assert design_sha == design["design_sha256"]
    assert design["schema_version"] == "three-action-utility-design-v2"
    assert design["round"] == 34
    assert design["design_revision"] == 4
    assert [profile["profile"] for profile in profiles] == [
        "conservative",
        "regular",
        "aggressive",
    ]


def test_round34_runner_rejects_unhashed_design_change(tmp_path: Path) -> None:
    payload = json.loads(DESIGN.read_text(encoding="utf-8"))
    payload["model"]["probability_derivation"][
        "semantic_aliasing_between_action_class_and_side_profit_probabilities_permitted"
    ] = True
    changed = tmp_path / DESIGN.name
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="design hash"):
        load_round34_design(changed)


def test_round34_architecture_diagnostics_cover_both_probability_systems() -> None:
    targets, ensemble = _diagnostic_bundle()
    design, _design_sha, _profiles = load_round34_design(DESIGN)
    gates = design["acceptance_gates"]["calibration_architecture"]
    diagnostics = _calibration_architecture_diagnostics(targets, ensemble)

    assert diagnostics["opportunity_rows"] == 10
    assert diagnostics["abstain_rows"] == 10
    assert diagnostics["opportunity_auc"] == 1.0
    assert diagnostics["conditional_direction_auc"] == 1.0
    assert diagnostics["side_profit_auc"] == 1.0
    assert diagnostics["side_profit_brier_to_base_rate_ratio"] < 1.0
    assert diagnostics["multiclass_log_loss_to_class_prior_ratio"] < 1.0
    assert (
        diagnostics["selected_action"]["top_rows"]["100"]["mean_stress_net_bps"] > 0.0
    )
    assert _architecture_gate_reasons(diagnostics, gates) == []


def test_round34_positive_exact_tie_is_abstain_but_both_sides_are_profitable() -> None:
    targets, ensemble = _diagnostic_bundle()
    targets.stress_long_net_bps[0] = 5.0
    targets.stress_short_net_bps[0] = 5.0

    diagnostics = _calibration_architecture_diagnostics(targets, ensemble)

    assert diagnostics["opportunity_rows"] == 9
    assert diagnostics["abstain_rows"] == 11
    assert diagnostics["action_class_support"]["abstain_rows"] == 11
    assert diagnostics["side_profit_positive_rows"] == 11


@pytest.mark.parametrize(
    ("metric", "value", "reason"),
    [
        ("opportunity_auc", 0.649, "opportunity_auc_gate_failed"),
        (
            "conditional_direction_auc",
            0.549,
            "conditional_direction_auc_gate_failed",
        ),
        ("side_profit_auc", 0.549, "side_profit_auc_gate_failed"),
        (
            "side_profit_brier_to_base_rate_ratio",
            1.001,
            "side_profit_brier_gate_failed",
        ),
        (
            "multiclass_log_loss_to_class_prior_ratio",
            1.001,
            "multiclass_log_loss_gate_failed",
        ),
    ],
)
def test_round34_architecture_probability_gates_fail_closed(
    metric: str,
    value: float,
    reason: str,
) -> None:
    targets, ensemble = _diagnostic_bundle()
    design, _design_sha, _profiles = load_round34_design(DESIGN)
    gates = design["acceptance_gates"]["calibration_architecture"]
    diagnostics = _calibration_architecture_diagnostics(targets, ensemble)
    diagnostics[metric] = value

    assert reason in _architecture_gate_reasons(diagnostics, gates)


def test_round34_architecture_rejects_nonfinite_metric() -> None:
    targets, ensemble = _diagnostic_bundle()
    design, _design_sha, _profiles = load_round34_design(DESIGN)
    gates = design["acceptance_gates"]["calibration_architecture"]
    diagnostics = _calibration_architecture_diagnostics(targets, ensemble)
    diagnostics["side_profit_auc"] = float("nan")

    assert "side_profit_auc_gate_failed" in _architecture_gate_reasons(
        diagnostics,
        gates,
    )


def test_round34_policy_adapter_does_not_alias_probability_semantics() -> None:
    _targets, ensemble = _diagnostic_bundle()
    policy_input = as_selective_action_ensemble(ensemble)

    np.testing.assert_array_equal(
        policy_input.action_values.long_profitable_probability,
        ensemble.action_values.long_profitable_probability,
    )
    assert not np.array_equal(
        policy_input.action_values.long_profitable_probability,
        ensemble.long_action_probability_mean,
    )
    np.testing.assert_array_equal(
        policy_input.opportunity_probability_mean,
        ensemble.long_action_probability_mean + ensemble.short_action_probability_mean,
    )
