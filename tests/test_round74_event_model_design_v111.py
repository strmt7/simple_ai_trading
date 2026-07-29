from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_PROMOTION_TASK_METRICS,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v111.json"
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


def test_round74_v111_binds_task_noninferiority_implementation() -> None:
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


def test_round74_v111_blocks_hidden_task_loss_without_claiming_edge() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    schemas = design["schema_contract"]
    verification = design["verification"]

    assert schemas["event_training"] == ROUND74_EVENT_TRAINING_SCHEMA_VERSION
    assert (
        schemas["event_pretest_policy"] == ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
    )
    assert tuple(delta["promotion_tasks"]) == tuple(
        task_name for task_name, _metric_name in ROUND74_EVENT_PROMOTION_TASK_METRICS
    )
    assert delta["promotion_task_count"] == len(ROUND74_EVENT_PROMOTION_TASK_METRICS)
    assert delta["task_deterioration_may_be_offset_by_other_tasks"] is False
    assert (
        delta["every_paired_run_symbol_horizon_task_noninferior_within_numerical_floor"]
        is True
    )
    assert delta["additional_forward_passes"] == 0
    assert delta["additional_backward_passes"] == 0
    assert delta["training_objective_changed"] is False
    assert verification["focused_tests_passed"] == 53
    assert verification[
        "hidden_sol_net_payoff_task_degradation_counterexample_rejected"
    ]
    assert verification["task_noninferiority_gate_is_decisive_in_counterexample"]
    assert verification["representative_training_performed"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
