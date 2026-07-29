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
    / "round-074-event-sequence-model-design-v119.json"
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


def test_round74_v119_binds_joint_ai_context_implementation() -> None:
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


def test_round74_v119_preserves_ai_authority_and_evidence_boundaries() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    context = design["ai_joint_probability_context"]
    screen = design["semantic_screen_delta"]
    verification = design["verification"]

    assert context["request_schema_version"] == "round-074-ai-review-request-v6"
    assert context["prompt_payload_schema_version"] == (
        "round-074-ai-prompt-payload-v9"
    )
    assert context["probability_sum_required"] == 1.0
    assert context["directional_margin_precomputed_for_ai"] is True
    assert context["ai_may_switch_side"] is False
    assert context["ai_may_increase_size"] is False
    assert context["ai_may_set_leverage"] is False
    assert context["ai_may_submit_or_cancel_orders"] is False
    assert screen["new_case_count"] == 10
    assert screen["side_mirror_consistency_required"] is True
    assert verification["unique_tests_passed"] == 189
    assert verification["real_local_ai_inference_executed_for_v119"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["drawdown_claim"] is False
    assert set(design["authority"].values()) == {False}
