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
    / "round-074-event-sequence-model-design-v120.json"
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


def test_round74_v120_binds_epistemic_and_ai_preload_evidence() -> None:
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
    preload = design["fin_r1_preload_evidence"]
    preload_path = ROOT / preload["path"]
    assert (
        preload["file_sha256"] == hashlib.sha256(preload_path.read_bytes()).hexdigest()
    )
    report = json.loads(preload_path.read_text(encoding="ascii"))
    assert preload["report_sha256"] == report["report_sha256"]
    assert preload["model_manifest_sha256"] == report["model_manifest_sha256"]
    assert preload["model_artifact_sha256"] == report["model_artifact_sha256"]
    assert preload["failure_message"] == report["failure_message"]


def test_round74_v120_preserves_policy_and_evidence_boundaries() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    telemetry = design["epistemic_telemetry_contract"]
    preconditions = design["epistemic_policy_challenge_preconditions"]
    preload = design["fin_r1_preload_evidence"]
    verification = design["verification"]

    assert telemetry["production_peer_count"] == 3
    assert telemetry["additional_model_forward_passes"] == 0
    assert telemetry["candidate_eligibility_changed"] is False
    assert telemetry["candidate_ranking_changed"] is False
    assert telemetry["position_size_changed"] is False
    assert telemetry["risk_policy_changed"] is False
    assert preconditions["status"] == "not_started"
    assert preconditions["aggregate_only_promotion_permitted"] is False
    assert preconditions["sealed_test_use_permitted"] is False
    assert preload["complete"] is False
    assert preload["screen_case_count_completed"] == 0
    assert preload["minimum_free_system_ram_requirement_relaxed"] is False
    assert preload["model_absent_after_cleanup_verified"] is True
    assert verification["unique_tests_passed"] == 168
    assert design["evidence_boundary"]["epistemic_risk_coverage_evaluated"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert set(design["authority"].values()) == {False}
