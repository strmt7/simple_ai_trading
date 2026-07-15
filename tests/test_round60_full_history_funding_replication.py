from __future__ import annotations

from pathlib import Path

from tools.run_round59_funding_persistence_feasibility import _canonical_sha256
from tools.run_round60_full_history_funding_replication import (
    ROOT,
    _gate_passed,
    _validate_design,
)


DESIGN = (
    ROOT
    / "docs/model-research/action-value/round-060-full-history-funding-replication-design.json"
)


def test_round60_reuses_every_round59_decision_rule_by_hash() -> None:
    design, protocol = _validate_design(DESIGN)

    assert design["source_contract"]["ranges_by_symbol"] == {
        "BTCUSDT": {
            "start_period": "2020-01",
            "end_period": "2026-06",
            "period_count": 78,
        },
        "ETHUSDT": {
            "start_period": "2020-01",
            "end_period": "2026-06",
            "period_count": 78,
        },
        "SOLUSDT": {
            "start_period": "2020-09",
            "end_period": "2026-06",
            "period_count": 70,
        },
    }
    for name, expected in design["protocol_reuse_contract"]["reused_sections"].items():
        assert _canonical_sha256(protocol[name]) == expected


def test_round60_gate_rejects_small_samples_without_reading_null_metrics() -> None:
    metrics = {
        "episodes": 39,
        "cost_comparisons": {
            "stress_four_leg": {
                "positive_net_reference_fraction": None,
                "median_net_reference_bps": None,
                "bootstrap_lower_95_mean_net_reference_bps": None,
            }
        },
    }
    gate = {
        "minimum_nonoverlapping_episodes_per_symbol": 40,
        "minimum_stress_net_positive_fraction": 0.55,
        "median_stress_net_bps_strictly_above": 0.0,
        "bootstrap_lower_95_mean_stress_net_bps_strictly_above": 0.0,
    }

    assert _gate_passed(metrics, gate) is False


def test_round60_gate_requires_every_precommitted_threshold() -> None:
    gate = {
        "minimum_nonoverlapping_episodes_per_symbol": 40,
        "minimum_stress_net_positive_fraction": 0.55,
        "median_stress_net_bps_strictly_above": 0.0,
        "bootstrap_lower_95_mean_stress_net_bps_strictly_above": 0.0,
    }
    passing = {
        "episodes": 40,
        "cost_comparisons": {
            "stress_four_leg": {
                "positive_net_reference_fraction": 0.55,
                "median_net_reference_bps": 0.01,
                "bootstrap_lower_95_mean_net_reference_bps": 0.01,
            }
        },
    }

    assert _gate_passed(passing, gate) is True
    for field in (
        "positive_net_reference_fraction",
        "median_net_reference_bps",
        "bootstrap_lower_95_mean_net_reference_bps",
    ):
        failing = {
            "episodes": passing["episodes"],
            "cost_comparisons": {
                "stress_four_leg": dict(passing["cost_comparisons"]["stress_four_leg"])
            },
        }
        failing["cost_comparisons"]["stress_four_leg"][field] = -0.01
        assert _gate_passed(failing, gate) is False


def test_round60_design_path_is_repository_local() -> None:
    assert isinstance(DESIGN, Path)
    assert DESIGN.is_file()
