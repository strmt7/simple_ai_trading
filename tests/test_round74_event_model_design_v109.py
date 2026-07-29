from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v109.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _normalized_source_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v109_binds_quantile_skill_implementation() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == hashlib.sha256(base_path.read_bytes()).hexdigest()
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )
    implementation_commit = design["implementation_git_commit"]
    for source in design["source_binding"].values():
        if not isinstance(source, dict):
            continue
        assert source["sha256"] == _normalized_source_sha256(
            implementation_commit,
            source["path"],
        )


def test_round74_v109_requires_quantile_skill_without_claiming_edge() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    verification = design["verification"]

    assert delta["sealed_evaluation_schema_version"] == (
        "round-074-sealed-evaluation-v20"
    )
    assert delta["temperature_calibration_schema_version"] == (
        "round-074-temperature-calibration-v5"
    )
    assert delta["no_information_quantile_baseline_schema_version"] == (
        "round-074-no-information-quantile-baseline-v1"
    )
    assert tuple(delta["predictive_tasks"]) == (
        "positive_payoff",
        "adverse_selection",
        "regime_unpredictability",
        "net_payoff_quantiles",
        "maximum_adverse_excursion_quantiles",
    )
    assert delta["bootstrap_draws"] == 10_000
    assert delta["familywise_alpha"] == 0.05
    assert delta["familywise_task_count"] == 5
    assert delta["per_task_one_sided_alpha"] == 0.01
    assert delta["capture_run_busy_row_duplication_changes_baseline"] is False
    assert delta["every_calibration_run_symbol_group_requires_eligible_support"] is True
    assert delta["sealed_test_labels_used_for_quantile_baseline_fit"] is False
    assert delta["quantile_baseline_is_probability_calibration_hash_bound"] is True
    assert delta["financial_gate_remains_independent_and_mandatory"] is True
    assert delta["ai_overlay_may_rescue_predictive_gate_failure"] is False
    assert verification["focused_tests_passed"] == 68
    assert verification["capture_run_baseline_duplication_invariance_passed"] is True
    assert verification["representative_training_performed"] is False
    assert verification["representative_predictive_skill_evaluated"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
