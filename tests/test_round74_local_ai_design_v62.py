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
    / "round-074-local-ai-review-design-v62.json"
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


def test_round74_local_ai_v62_preserves_dependence_and_claim_boundaries() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-local-ai-review-design-v62"

    base = design["base_design_binding"]
    base_path = REPOSITORY / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )

    commit = design["implementation_git_commit"]
    source = design["source_binding"]
    for prefix in (
        "sealed_evaluator",
        "sealed_ledger_test",
        "historical_ai_design_test",
    ):
        assert source[f"{prefix}_sha256"] == _git_blob_sha256(
            commit,
            source[f"{prefix}_path"],
        )
    assert source["sealed_evaluator_schema_version"] == (
        "round-074-sealed-evaluation-v15"
    )

    delta = design["declared_delta"]
    assert delta["iid_capture_run_bootstrap_removed"] is True
    assert delta["circular_stationary_bootstrap_added"] is True
    assert delta["chronological_capture_run_order_preserved"] is True
    assert delta["mean_block_length_uses_target_or_pnl_values"] is False
    assert delta["two_ai_model_bonferroni_gate_changed"] is False
    assert delta["sealed_test_consumed"] is False

    contract = design["sealed_qualification_contract"]
    assert contract["ai_model_count"] == 2
    assert contract["ai_action_authority"] == "retain_reduce_or_veto_only"
    assert contract["expected_test_capture_runs"] == (
        "all admitted segments in the immutable sealed test population"
    )
    assert contract["bootstrap_method"] == "circular_stationary_bootstrap"
    assert contract["iid_capture_run_resampling_permitted"] is False
    assert contract["bootstrap_block_length_selected_without_outcomes"] is True
    assert (
        contract[
            "two_model_bonferroni_positive_stationary_run_block_lower_bound_required"
        ]
        is True
    )
    assert contract["every_capture_run_delta_net_bps_minimum"] == -1e-12
    assert contract["every_observed_symbol_horizon_delta_net_bps_minimum"] == -1e-12
    assert contract["promotion_authority"] is False
    assert contract["trading_authority"] is False

    boundary = design["assumption_boundary"]
    assert (
        boundary[
            "weak_stationarity_and_weak_dependence_required_for_bootstrap_interpretation"
        ]
        is True
    )
    assert boundary["stationarity_proven_by_implementation"] is False
    assert boundary["block_length_optimality_claim"] is False
    assert boundary["independence_between_adjacent_capture_runs_assumed"] is False
    assert boundary["bootstrap_lower_bound_alone_can_establish_ai_uplift"] is False
    assert len(design["research_basis"]) == 4
    assert design["verification"]["real_sealed_test_accessed"] is False
    assert design["evidence_limits"]["ai_model_evaluated_for_market_uplift"] is False
    assert not any(design["status"].values())
