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
    / "round-074-event-sequence-model-design-v106.json"
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256_at(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v106_binds_gradient_health_implementation() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )
    implementation_commit = design["implementation_git_commit"]
    for source in design["source_binding"].values():
        assert source["sha256"] == _source_sha256_at(
            implementation_commit,
            source["path"],
        )


def test_round74_v106_records_telemetry_without_optimizer_or_edge_claims() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    schemas = design["schema_contract"]
    verification = design["verification"]

    assert delta["additional_forward_passes_per_optimizer_step"] == 0
    assert delta["additional_backward_passes_per_optimizer_step"] == 0
    assert delta["additional_gradient_norm_traversals_per_optimizer_step"] == 0
    assert delta["optimizer_parameter_updates_changed"] is False
    assert delta["gradnorm_enabled"] is False
    assert delta["pcgrad_enabled"] is False
    assert delta["nash_mtl_enabled"] is False
    assert delta["per_task_gradient_cosines_measured"] is False
    assert delta["gradient_conflict_claim"] is False
    assert schemas == {
        "event_training": "round-074-event-training-v31",
        "event_pretest_policy": "round-074-event-pretest-policy-v30",
        "event_training_preflight_run": "round-074-event-training-preflight-run-v8",
        "event_training_directml_preflight_evidence": (
            "round-074-event-training-directml-preflight-evidence-v20"
        ),
    }
    assert verification["focused_tests_passed"] == 104
    assert verification["directml_preflight_executed_for_this_delta"] is False
    assert verification["representative_training_performed"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert set(design["authority"].values()) == {False}
