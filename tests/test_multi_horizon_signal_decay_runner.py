from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_multi_horizon_signal_decay import (
    _EXPECTED_HORIZONS,
    _EXPECTED_SIGNALS,
    _REQUIRED_BOUND_PATHS,
    _canonical_sha256,
    _daily_summary,
    _half_life,
    _memory_evidence,
    load_signal_decay_binding,
    load_signal_decay_design,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-036-multi-horizon-signal-decay-design.json"
)
BINDING_V1 = DESIGN.with_name("round-036-signal-decay-execution-binding-v1.json")
BINDING = DESIGN.with_name("round-036-signal-decay-execution-binding.json")


def test_round36_runner_loads_the_exact_frozen_budget() -> None:
    design, design_sha = load_signal_decay_design(DESIGN)

    assert design_sha == (
        "276e0c169b2bd24ce87843b66892c4139daffdd581776457d068d4b63b61727e"
    )
    assert tuple(item["name"] for item in design["signals"]) == _EXPECTED_SIGNALS
    assert tuple(design["horizons_seconds"]) == _EXPECTED_HORIZONS
    assert len(_REQUIRED_BOUND_PATHS) == 41
    assert "src/simple_ai_trading/microstructure_signal_decay.py" in (
        _REQUIRED_BOUND_PATHS
    )
    assert "tools/run_multi_horizon_signal_decay.py" in _REQUIRED_BOUND_PATHS


def test_round36_runner_rejects_design_drift_before_execution(tmp_path) -> None:
    payload = json.loads(DESIGN.read_text(encoding="utf-8"))
    payload["horizons_seconds"] = [5, 15, 30]
    drifted = tmp_path / DESIGN.name
    drifted.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash or identity"):
        load_signal_decay_design(drifted)


def test_round36_v1_binding_remains_canonically_verifiable() -> None:
    binding = json.loads(BINDING_V1.read_text(encoding="utf-8"))
    canonical = dict(binding)
    binding_sha = canonical.pop("binding_sha256")

    assert binding_sha == (
        "1fdeac5134715ff368047e065f7aefff8b0d159477e485524b771083edda3405"
    )
    assert binding_sha == _canonical_sha256(canonical)
    assert binding["implementation"]["commit"] == (
        "9f69a5385237dc34bbea802518879374d2df33cd"
    )
    assert len(binding["implementation"]["files"]) == 41
    assert binding["authority"] == {
        "post_hoc_consumed_data_diagnostic_only": True,
        "model_training_permitted": False,
        "model_candidate_permitted": False,
        "promotion_permitted": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def test_round36_current_binding_matches_hardened_implementation() -> None:
    _design, design_sha = load_signal_decay_design(DESIGN)
    binding, binding_sha = load_signal_decay_binding(
        BINDING,
        design_path=DESIGN,
        design_sha256=design_sha,
    )

    assert binding_sha == (
        "76bf3a8c39e658875c49bf91230c50322b3d37584492a05ee7aab4d01bcaead8"
    )
    assert binding["implementation"]["commit"] == (
        "3c16bed56370c2dbf393fbb8dd19f54007419ee0"
    )
    assert len(binding["implementation"]["files"]) == 41


def test_daily_summary_reports_defined_support_without_imputation() -> None:
    records = [
        {"weighted_roc_auc": 0.55},
        {"weighted_roc_auc": None},
        {"weighted_roc_auc": 0.45},
        {"weighted_roc_auc": 0.60},
        {"weighted_roc_auc": 0.50},
    ]

    summary = _daily_summary(records)

    assert summary == {
        "days": 5,
        "days_with_defined_auc": 4,
        "days_above_chance": 2,
        "weighted_auc_minimum": 0.45,
        "weighted_auc_median": 0.525,
        "weighted_auc_standard_deviation": pytest.approx(0.05590169943749474),
    }


def _decay_results(
    aucs: list[float],
    *,
    days_above_chance: int,
) -> list[dict[str, object]]:
    return [
        {
            "horizon_seconds": horizon,
            "direction": {
                "weighted_roc_auc": auc,
                "spearman_information_coefficient": auc - 0.5,
            },
            "daily_summary": {"days_above_chance": days_above_chance},
        }
        for horizon, auc in zip(_EXPECTED_HORIZONS, aucs, strict=True)
    ]


def test_half_life_requires_frozen_strength_daily_and_monotonic_conditions() -> None:
    measurable = _half_life(
        _decay_results(
            [0.54, 0.535, 0.52, 0.515, 0.51, 0.505, 0.50],
            days_above_chance=4,
        )
    )
    unstable = _half_life(
        _decay_results(
            [0.54, 0.535, 0.52, 0.515, 0.51, 0.505, 0.50],
            days_above_chance=3,
        )
    )

    assert measurable["half_life_status"] == "measurable_on_consumed_role_only"
    assert measurable["half_life_seconds"] == 30.0
    assert measurable["earliest_peak_horizon_seconds"] == 5
    assert unstable["half_life_status"] == ("no_measurable_half_life_on_consumed_role")
    assert unstable["half_life_seconds"] is None


def test_process_memory_evidence_uses_a_supported_host_counter() -> None:
    evidence = _memory_evidence()

    assert evidence["source"] in {
        "windows_process_memory_counters",
        "getrusage",
    }
    assert int(evidence["peak_working_set_bytes"]) > 0
