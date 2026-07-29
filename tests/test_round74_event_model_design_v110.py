from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_sealed_evaluation import (
    ROUND74_SEALED_BOOTSTRAP_DRAWS,
    ROUND74_SEALED_EVALUATION_SCHEMA_VERSION,
    ROUND74_SEALED_FAMILYWISE_ALPHA,
    ROUND74_SEALED_PREDICTIVE_COMPARISON_COUNT,
    ROUND74_SEALED_PREDICTIVE_TASKS,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_SYMBOLS,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v110.json"
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


def test_round74_v110_binds_symbol_skill_implementation() -> None:
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


def test_round74_v110_requires_every_symbol_without_claiming_edge() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    verification = design["verification"]

    assert delta["sealed_evaluation_schema_version"] == (
        ROUND74_SEALED_EVALUATION_SCHEMA_VERSION
    )
    assert tuple(delta["predictive_tasks"]) == ROUND74_SEALED_PREDICTIVE_TASKS
    assert tuple(delta["symbols"]) == ROUND74_EVENT_SYMBOLS
    assert delta["bootstrap_draws_per_comparison"] == ROUND74_SEALED_BOOTSTRAP_DRAWS
    assert delta["familywise_alpha"] == ROUND74_SEALED_FAMILYWISE_ALPHA
    assert delta["familywise_predictive_comparison_count"] == (
        ROUND74_SEALED_PREDICTIVE_COMPARISON_COUNT
    )
    assert delta["per_comparison_one_sided_alpha"] == (
        ROUND74_SEALED_FAMILYWISE_ALPHA / ROUND74_SEALED_PREDICTIVE_COMPARISON_COUNT
    )
    assert delta["symbol_specific_failure_may_be_offset_by_other_symbols"] is False
    assert delta["every_expected_test_capture_run_requires_symbol_evidence"] is True
    assert delta["financial_gate_remains_independent_and_mandatory"] is True
    assert delta["ai_overlay_may_rescue_predictive_gate_failure"] is False
    assert verification["focused_tests_passed"] == 70
    assert verification["pooled_payoff_gate_passes_masking_counterexample"] is True
    assert (
        verification["unskilled_sol_payoff_symbol_gate_rejects_masking_counterexample"]
        is True
    )
    assert verification["representative_training_performed"] is False
    assert verification["representative_predictive_skill_evaluated"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
