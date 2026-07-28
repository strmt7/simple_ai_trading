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
    / "round-074-event-sequence-model-design-v91.json"
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


def test_round74_v91_binds_state_conditioned_flow_after_feature_selection() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v91"

    base = design["base_design"]
    base_path = REPOSITORY / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert base["design_sha256"] == json.loads(
        base_path.read_text(encoding="ascii")
    )["design_sha256"]

    source = design["source_binding"]
    commit = design["implementation_git_commit"]
    for prefix in (
        "feature_partition",
        "event_model",
        "event_training",
        "model_test",
        "training_test",
        "contract_test",
    ):
        assert source[f"{prefix}_sha256"] == _git_blob_sha256(
            commit,
            source[f"{prefix}_path"],
        )
    directml_path = REPOSITORY / source["directml_evidence_path"]
    assert source["directml_evidence_file_sha256"] == _file_sha256(directml_path)
    directml_evidence = json.loads(directml_path.read_text(encoding="ascii"))
    assert directml_evidence["implementation_git_commit"] == commit
    assert directml_evidence["probe"]["cpu_fallback_warning_count"] == 0
    assert directml_evidence["operator_harness_history"]["attempt_count"] == 2
    assert directml_evidence["operator_harness_history"][
        "first_attempt_status"
    ] == "failed_cpu_fallback_gate"
    assert directml_evidence["operator_harness_history"]["second_attempt_status"] == (
        "passed"
    )

    partition = design["feature_partition_contract"]
    assert (
        partition["market_state_feature_count"]
        + partition["order_flow_feature_count"]
        + partition["clock_and_intraday_feature_count"]
        == partition["feature_count"]
        == 75
    )
    assert partition["partition_is_disjoint_and_complete"] is True
    assert partition["masked_order_flow_cannot_activate_interaction"] is True

    interaction = design["state_conditioned_flow_contract"]
    assert interaction["initial_flow_multiplier"] == 1.0
    assert interaction["weight_initialization"] == 0.0
    assert interaction["bias_initialization"] == 0.0
    assert interaction["initial_output_exactly_matches_equal_seed_incumbent"] is True
    assert interaction["non_order_flow_columns_are_unchanged"] is True
    assert interaction["additional_parameter_count_per_peer"] == 1054
    assert interaction["state_to_flow_weight_shape"] == [31, 33]
    assert interaction["state_to_flow_bias_shape"] == [31]

    selection = design["selection_contract"]
    assert selection["stage_order"] == [
        "select_base_architecture_on_market_state_clock_neutral",
        "challenge_market_state_with_clock",
        "challenge_selected_clock_view_with_order_flow",
        "challenge_selected_order_flow_view_with_state_conditioning",
        "challenge_random_initialization_with_causal_pretraining",
        "fit_calibration_after_all_development_selection_is_frozen",
        "evaluate_sealed_test_once",
    ]
    assert selection["interaction_stage_requires_selected_order_flow"] is True
    assert selection["interaction_stage_skipped_when_order_flow_rejected"] is True
    assert selection["strict_mean_proper_loss_improvement_required"] is True
    assert selection["every_paired_run_must_be_noninferior_within_numerical_floor"]
    assert selection[
        "every_run_symbol_horizon_subgroup_must_be_noninferior_within_numerical_floor"
    ]
    assert selection["unconditioned_incumbent_is_default_on_any_gate_failure"]
    assert selection["backtest_pnl_used_for_interaction_selection"] is False
    assert selection["sealed_test_used_for_interaction_selection"] is False

    assert len(design["primary_source_review"]) == 4
    assert all(
        source_review["transfer_limit"]
        for source_review in design["primary_source_review"]
    )
    verification = design["verification"]
    assert verification["total_connected_test_count"] == 114
    assert verification["design_integrity_test_count"] == 1
    assert verification["directml_source_bound_test_passed"] is True
    assert verification["hidden_subgroup_degradation_rejected"] is True
    assert verification["full_representative_training_performed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert not any(design["authority"].values())
