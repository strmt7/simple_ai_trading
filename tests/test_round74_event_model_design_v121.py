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
    / "round-074-event-sequence-model-design-v121.json"
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


def test_round74_v121_is_canonical_and_source_bound() -> None:
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
    commit = design["implementation_git_commit"]
    for source in design["source_binding"].values():
        assert source["sha256"] == _normalized_source_sha256(
            commit,
            source["path"],
        )


def test_round74_v121_preserves_policy_and_evidence_boundaries() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    contract = design["epistemic_risk_coverage_contract"]
    integration = design["development_integration"]
    verification = design["verification"]
    campaign = design["campaign_observation"]
    boundary = design["evidence_boundary"]

    assert contract["status"] == (
        "implemented_waiting_for_representative_policy_selection_runs"
    )
    assert contract["expected_capture_runs"] == 6
    assert contract["aggregate_metric_count"] == 5
    assert contract["conditional_metric_count"] == 216
    assert contract["total_metric_count"] == 221
    assert contract["aggregate_only_promotion_permitted"] is False
    assert contract["automatic_policy_gate_enabled"] is False
    assert integration["additional_model_forward_passes"] == 0
    assert integration["candidate_derivation_reads_report"] is False
    assert integration["action_policy_selection_reads_report"] is False
    assert integration["ai_review_reads_report"] is False
    assert integration["execution_reads_report"] is False
    assert verification["focused_tests_passed"] == 20
    assert verification["representative_policy_selection_evaluation_run"] is False
    assert campaign["admitted_slots_confirmed"] == 41
    assert campaign["latest_admitted_slot"] == 51
    assert campaign["latest_reconnect_count"] == 0
    assert campaign["latest_credentials_used"] is False
    assert campaign["latest_orders_submitted"] is False
    assert boundary["epistemic_evaluator_implemented"] is True
    assert boundary["epistemic_evaluator_integrated"] is True
    assert boundary["representative_epistemic_risk_coverage_evaluated"] is False
    assert boundary["policy_challenge_eligible"] is False
    assert boundary["uncertainty_policy_use_enabled"] is False
    assert boundary["financial_edge_tested"] is False
    assert boundary["profitability_claim"] is False
    assert set(design["authority"].values()) == {False}
