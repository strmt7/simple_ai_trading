from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_architecture import (
    ActionValueEnsembleBatch,
)
from simple_ai_trading.microstructure_shared_action_lightgbm import (
    SharedActionEnsembleBatch,
)
from tools.run_shared_action_viability import (
    _forecast_gate_reasons,
    _selected_action_diagnostics,
    load_round32_design,
    load_round32_execution_binding,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-032-shared-action-value-viability-design.json"
)
BINDING = DESIGN.with_name("round-032-execution-binding.json")


def _diagnostic_bundle() -> tuple[SimpleNamespace, SharedActionEnsembleBatch]:
    rows = 10
    endpoints = np.arange(rows, dtype=np.int64)
    magnitude = np.arange(1.0, rows + 1.0)
    long_side = endpoints % 2 == 0
    long_actual = np.where(long_side, magnitude, -magnitude)
    short_actual = -long_actual
    long_prediction = long_actual * 0.8
    short_prediction = short_actual * 0.8
    standard_deviation = np.full(rows, 0.1, dtype=np.float64)
    action_values = ActionValueEnsembleBatch(
        endpoint_indexes=endpoints,
        long_mean_bps=long_prediction,
        short_mean_bps=short_prediction,
        long_epistemic_std_bps=standard_deviation,
        short_epistemic_std_bps=standard_deviation,
        long_profitable_probability=np.where(long_side, 0.9, 0.1),
        short_profitable_probability=np.where(long_side, 0.1, 0.9),
        long_lower_bps=long_prediction - 1.0,
        short_lower_bps=short_prediction - 1.0,
        long_upper_bps=long_prediction + 1.0,
        short_upper_bps=short_prediction + 1.0,
        long_positive_member_ratio=np.where(long_side, 1.0, 0.0),
        short_positive_member_ratio=np.where(long_side, 0.0, 1.0),
        member_count=3,
    )
    advantage = long_prediction - short_prediction
    ensemble = SharedActionEnsembleBatch(
        action_values=action_values,
        signed_advantage_mean_bps=advantage,
        signed_advantage_epistemic_std_bps=standard_deviation,
        advantage_long_member_ratio=np.where(long_side, 1.0, 0.0),
        advantage_short_member_ratio=np.where(long_side, 0.0, 1.0),
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


def test_round32_runner_loads_exact_frozen_design() -> None:
    design, design_sha, profiles = load_round32_design(DESIGN)

    assert design_sha == design["design_sha256"]
    assert design["design_revision"] == 4
    assert [profile["profile"] for profile in profiles] == [
        "conservative",
        "regular",
        "aggressive",
    ]


def test_round32_execution_binding_matches_current_git_blobs() -> None:
    _design, design_sha, _profiles = load_round32_design(DESIGN)
    binding, binding_sha = load_round32_execution_binding(
        BINDING,
        design_path=DESIGN,
        design_sha256=design_sha,
    )

    assert binding_sha == binding["binding_sha256"]
    assert binding["implementation"]["commit"] == (
        "8212805ee76d327ccedc859247ea92acc42f3670"
    )
    assert len(binding["implementation"]["files"]) == 21


def test_round32_runner_rejects_unhashed_design_change(tmp_path: Path) -> None:
    payload = json.loads(DESIGN.read_text(encoding="utf-8"))
    payload["model"]["seeds"][0] = 31
    changed = tmp_path / DESIGN.name
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="design hash"):
        load_round32_design(changed)


def test_selected_action_diagnostics_and_frozen_gates_are_directional() -> None:
    targets, ensemble = _diagnostic_bundle()
    diagnostics = _selected_action_diagnostics(targets, ensemble)
    design, _design_sha, _profiles = load_round32_design(DESIGN)
    gates = design["acceptance_gates"]["distant_confirmation_forecast"]

    assert diagnostics["side_choice_auc"] == 1.0
    assert diagnostics["selected_action_pearson_information_coefficient"] > 0.99
    assert diagnostics["selected_action_spearman_information_coefficient"] > 0.99
    assert diagnostics["top_rows"]["500"]["mean_stress_net_bps"] > 0.0
    assert diagnostics["top_rows"]["500"]["long_share"] == 0.5
    assert _forecast_gate_reasons(diagnostics, gates) == []

    failed = dict(diagnostics)
    failed["side_choice_auc"] = 0.51
    assert "side_choice_auc_gate_failed" in _forecast_gate_reasons(failed, gates)
