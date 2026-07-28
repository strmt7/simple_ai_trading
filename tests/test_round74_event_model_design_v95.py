from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v95.json"
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


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_v95_binds_dependence_preserving_sealed_inference() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v95"

    base = design["base_design"]
    base_path = REPOSITORY / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )

    commit = design["implementation_git_commit"]
    source = design["source_binding"]
    for prefix in ("sealed_evaluator", "sealed_ledger_test"):
        assert source[f"{prefix}_sha256"] == _git_blob_sha256(
            commit,
            source[f"{prefix}_path"],
        )
    ai_path = REPOSITORY / source["local_ai_design_path"]
    assert source["local_ai_design_file_sha256"] == _file_sha256(ai_path)
    ai_design = json.loads(ai_path.read_text(encoding="ascii"))
    assert source["local_ai_design_sha256"] == ai_design["design_sha256"]
    assert ai_design["implementation_git_commit"] == commit

    delta = design["declared_delta"]
    assert delta["sealed_evaluation_schema_version"] == (
        "round-074-sealed-evaluation-v15"
    )
    assert delta["iid_capture_run_bootstrap_removed"] is True
    assert delta["circular_stationary_bootstrap_added"] is True
    assert delta["bootstrap_block_length_uses_test_outcomes"] is False
    assert (
        delta["base_ml_financial_gate_uses_dependence_preserving_lower_bound"] is True
    )
    assert delta["local_ai_uplift_gate_uses_dependence_preserving_lower_bound"] is True
    assert delta["model_training_or_promotion_changed"] is False
    assert delta["sealed_test_consumed"] is False

    contract = design["sealed_inference_contract"]
    assert contract["resampling_unit"] == "whole_chronological_capture_run"
    assert contract["iid_capture_run_resampling_permitted"] is False
    assert contract["stationary_bootstrap_circular_wraparound"] is True
    assert contract["same_resampling_method_used_for_baseline_and_ai_delta"] is True
    assert contract["all_existing_financial_and_subgroup_gates_preserved"] is True
    assert design["assumption_boundary"]["stationarity_claim"] is False
    assert design["assumption_boundary"]["optimal_block_length_claim"] is False
    assert design["verification"]["sealed_test_evaluated"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert not any(design["authority"].values())
