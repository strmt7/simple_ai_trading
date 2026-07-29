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
    / "round-074-event-sequence-model-design-v117.json"
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


def test_round74_v117_binds_joint_positive_outcome_score() -> None:
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


def test_round74_v117_uses_proper_observed_score_without_claiming_edge() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    loss = design["loss_contract"]
    delta = design["declared_delta"]
    verification = design["verification"]

    assert loss["jointly_eligible_score"] == (
        "negative_log_probability_of_observed_three_outcome_class"
    )
    assert loss["singly_eligible_score"] == (
        "binary_log_score_of_only_observed_directional_marginal"
    )
    assert loss["joint_target_invented_for_censored_side"] is False
    assert loss["training_and_promotion_metric_identical"] is True
    assert delta["event_model_schema_version"] == "round-074-event-payoff-model-v12"
    assert delta["new_metric_name"] == "positive_log_loss"
    assert delta["neither_outcome_scored_directly"] is True
    assert delta["eligible_direction_weighting_changed"] is False
    assert delta["parameter_counts_changed"] is False
    assert verification["unique_tests_passed"] == 184
    assert verification["directml_forward_backward_preflight_passed"] is True
    assert verification["representative_training_performed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["drawdown_claim"] is False
    assert set(design["authority"].values()) == {False}
