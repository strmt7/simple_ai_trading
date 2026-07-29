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
    / "round-074-event-sequence-model-design-v118.json"
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


def test_round74_v118_binds_joint_payoff_calibration_implementation() -> None:
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


def test_round74_v118_preserves_evidence_and_authority_boundaries() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    contract = design["calibration_contract"]
    delta = design["declared_delta"]
    verification = design["verification"]

    assert contract["schema_version"] == "round-074-temperature-calibration-v6"
    assert contract["read_compatible_prior_schema_version"] == (
        "round-074-temperature-calibration-v5"
    )
    assert contract["fully_observed_selection_score"] == (
        "joint_three_outcome_log_loss"
    )
    assert contract["singly_observed_selection_score"] == (
        "eligible_directional_marginal_binary_log_loss"
    )
    assert contract["unobserved_direction_target_synthesized"] is False
    assert delta["event_model_changed"] is False
    assert delta["event_training_changed"] is False
    assert delta["action_policy_changed"] is False
    assert verification["unique_tests_passed"] == 179
    assert verification["directml_calibration_preflight_passed"] is True
    assert verification["representative_training_performed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["drawdown_claim"] is False
    assert set(design["authority"].values()) == {False}
