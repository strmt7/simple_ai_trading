from __future__ import annotations

import numpy as np

from tools.diagnose_round33_failure import (
    _binary_metrics,
    _routing_diagnostics,
)
from tests.test_selective_action_viability_runner import _diagnostic_bundle


def test_round33_binary_diagnostic_reports_calibration_and_discrimination() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    metrics = _binary_metrics(labels, probabilities)

    assert metrics["rows"] == 4
    assert metrics["positive_rows"] == 2
    assert metrics["roc_auc"] == 1.0
    assert 0.0 < metrics["brier_score"] < 0.1
    assert sum(row["rows"] for row in metrics["reliability_bins"]) == 4


def test_round33_routing_diagnostic_compares_seven_frozen_outputs() -> None:
    targets, ensemble = _diagnostic_bundle()
    diagnostics = _routing_diagnostics(
        ensemble,
        np.asarray(targets.stress_long_net_bps, dtype=np.float64),
        np.asarray(targets.stress_short_net_bps, dtype=np.float64),
    )

    assert len(diagnostics) == 7
    assert {row["routing"] for row in diagnostics} == {
        "action_value_mean_minus_one_epistemic_std",
        "opportunity_probability_with_conditional_direction_side",
        "joint_action_probability",
        "joint_action_probability_minus_abstain_probability",
        "opportunity_times_conditional_direction_confidence",
        "conditional_direction_confidence",
        "action_value_direction_consensus_only",
    }
    for diagnostic in diagnostics:
        assert diagnostic["active_rows"] > 0
        assert len(diagnostic["top_rows"]) == 4
        assert diagnostic["top_rows"][0]["actual_rows"] == diagnostic["active_rows"]
        assert np.isfinite(diagnostic["side_choice_auc"])
