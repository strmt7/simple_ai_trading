from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_architecture import (
    ActionValueEnsembleBatch,
)
from simple_ai_trading.microstructure_selective_action_lightgbm import (
    SelectiveActionEnsembleBatch,
)
from tools.run_selective_action_viability import (
    _architecture_gate_reasons,
    _calibration_architecture_diagnostics,
    load_round33_design,
    load_round33_execution_binding,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-033-selective-action-design.json"
)
BINDING = DESIGN.with_name("round-033-execution-binding.json")


def _diagnostic_bundle() -> tuple[SimpleNamespace, SelectiveActionEnsembleBatch]:
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
    opportunity_probability = np.where(opportunity, 0.9, 0.1)
    conditional_long = np.where(long_preferred, 0.9, 0.1)
    long_probability = opportunity_probability * conditional_long
    short_probability = opportunity_probability * (1.0 - conditional_long)
    long_prediction = np.where(long_preferred, magnitude * 0.8, -magnitude * 0.8)
    short_prediction = -long_prediction
    standard_deviation = np.full(rows, 0.1, dtype=np.float64)
    action = ActionValueEnsembleBatch(
        endpoint_indexes=endpoints,
        long_mean_bps=long_prediction,
        short_mean_bps=short_prediction,
        long_epistemic_std_bps=standard_deviation,
        short_epistemic_std_bps=standard_deviation,
        long_profitable_probability=long_probability,
        short_profitable_probability=short_probability,
        long_lower_bps=long_prediction - 1.0,
        short_lower_bps=short_prediction - 1.0,
        long_upper_bps=long_prediction + 1.0,
        short_upper_bps=short_prediction + 1.0,
        long_positive_member_ratio=np.where(long_preferred, 1.0, 0.0),
        short_positive_member_ratio=np.where(long_preferred, 0.0, 1.0),
        member_count=3,
    )
    opportunity_members = np.repeat(opportunity_probability[None, :], 3, axis=0)
    direction_members = np.repeat(conditional_long[None, :], 3, axis=0)
    ensemble = SelectiveActionEnsembleBatch(
        action_values=action,
        opportunity_probability_mean=opportunity_probability,
        opportunity_probability_std=np.zeros(rows),
        conditional_long_probability_mean=conditional_long,
        conditional_long_probability_std=np.zeros(rows),
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


def test_round33_runner_loads_exact_frozen_design() -> None:
    design, design_sha, profiles = load_round33_design(DESIGN)

    assert design_sha == design["design_sha256"]
    assert design["round"] == 33
    assert design["design_revision"] == 1
    assert [profile["profile"] for profile in profiles] == [
        "conservative",
        "regular",
        "aggressive",
    ]


def test_round33_runner_rejects_unhashed_design_change(tmp_path: Path) -> None:
    payload = json.loads(DESIGN.read_text(encoding="utf-8"))
    payload["conditional_direction_confidence"]["aggressive"] = 0.51
    changed = tmp_path / DESIGN.name
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="design hash"):
        load_round33_design(changed)


def test_round33_execution_binding_matches_current_git_blobs() -> None:
    _design, design_sha, _profiles = load_round33_design(DESIGN)
    binding, binding_sha = load_round33_execution_binding(
        BINDING,
        design_path=DESIGN,
        design_sha256=design_sha,
    )

    assert binding_sha == binding["binding_sha256"]
    assert binding["implementation"]["commit"] == (
        "6e1b41dbc2a9767caa8ef1deea6fc4652726ef2d"
    )
    assert len(binding["implementation"]["files"]) == 27


def test_round33_architecture_diagnostics_and_gates_are_selective() -> None:
    targets, ensemble = _diagnostic_bundle()
    design, _design_sha, _profiles = load_round33_design(DESIGN)
    gates = design["acceptance_gates"]["calibration_architecture"]
    diagnostics = _calibration_architecture_diagnostics(targets, ensemble)

    assert diagnostics["opportunity_rows"] == 10
    assert diagnostics["abstain_rows"] == 10
    assert diagnostics["opportunity_auc"] == 1.0
    assert diagnostics["conditional_direction_auc"] == 1.0
    assert diagnostics["selected_action"]["top_rows"]["100"][
        "mean_stress_net_bps"
    ] > 0.0
    assert _architecture_gate_reasons(diagnostics, gates) == []

    failed = dict(diagnostics)
    failed["conditional_direction_auc"] = 0.549
    assert "conditional_direction_auc_gate_failed" in _architecture_gate_reasons(
        failed,
        gates,
    )
