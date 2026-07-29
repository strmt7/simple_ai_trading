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
    / "round-074-event-sequence-model-design-v105.json"
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


def _source_sha256_candidates_at(commit: str, relative_path: str) -> set[str]:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    # This historical design bound the Windows worktree bytes before source
    # hash normalization was added to later artifacts.
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {
        hashlib.sha256(payload).hexdigest(),
        hashlib.sha256(payload.replace(b"\n", b"\r\n")).hexdigest(),
    }


def test_round74_v105_binds_deadline_aware_ai_queue_implementation() -> None:
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
        assert source["sha256"] in _source_sha256_candidates_at(
            implementation_commit,
            source["path"],
        )


def test_round74_v105_rejects_stale_queue_work_without_financial_claims() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    schemas = design["schema_contract"]
    verification = design["verification"]

    assert delta["queue_timeout_action"] == "reject_before_model_inference"
    assert delta["expired_request_invokes_injected_model_runner"] is False
    assert delta["expired_request_retained_as_paired_observation"] is True
    assert delta["expired_request_exposure_bps"] == 0
    assert delta["expired_request_counts_against_ai_qualification"] is True
    assert delta["candidate_models_are_independent_overlay_candidates"] is True
    assert delta["candidate_models_are_treated_as_concurrent_ensemble"] is False
    assert schemas == {
        "ai_review_panel": "round-074-ai-review-panel-v14",
        "ai_uplift_development": "round-074-ai-uplift-development-v13",
        "ai_pretest_qualification": "round-074-ai-pretest-qualification-v3",
        "ai_qualification_operator": "round-074-ai-qualification-operator-v3",
        "sealed_evaluation": "round-074-sealed-evaluation-v18",
        "segmented_qualified_development": (
            "round-074-segmented-qualified-development-v3"
        ),
        "segmented_development_run": "round-074-segmented-development-run-v3",
    }
    assert verification["focused_tests_passed"] == 158
    assert verification["representative_ai_reviews_executed"] is False
    assert verification["local_multibillion_parameter_model_invoked"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert set(design["authority"].values()) == {False}
