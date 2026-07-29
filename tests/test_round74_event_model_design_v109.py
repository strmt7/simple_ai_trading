from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_NO_INFORMATION_QUANTILE_BASELINE_SCHEMA_VERSION,
    ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_evaluation import (
    ROUND74_SEALED_BOOTSTRAP_DRAWS,
    ROUND74_SEALED_EVALUATION_SCHEMA_VERSION,
    ROUND74_SEALED_FAMILYWISE_ALPHA,
    ROUND74_SEALED_PREDICTIVE_TASKS,
)


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
        ROUND74_SEALED_EVALUATION_SCHEMA_VERSION
    )
    assert delta["temperature_calibration_schema_version"] == (
        ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
    )
    assert delta["no_information_quantile_baseline_schema_version"] == (
        ROUND74_NO_INFORMATION_QUANTILE_BASELINE_SCHEMA_VERSION
    )
    assert tuple(delta["predictive_tasks"]) == ROUND74_SEALED_PREDICTIVE_TASKS
    assert delta["bootstrap_draws"] == ROUND74_SEALED_BOOTSTRAP_DRAWS
    assert delta["familywise_alpha"] == ROUND74_SEALED_FAMILYWISE_ALPHA
    assert delta["familywise_task_count"] == len(ROUND74_SEALED_PREDICTIVE_TASKS)
    assert delta["per_task_one_sided_alpha"] == (
        ROUND74_SEALED_FAMILYWISE_ALPHA / len(ROUND74_SEALED_PREDICTIVE_TASKS)
    )
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
