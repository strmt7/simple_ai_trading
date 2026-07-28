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
    / "round-074-event-sequence-model-design-v92.json"
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


def test_round74_v92_isolates_adaptive_selection_and_preprocessing() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v92"

    base = design["base_design"]
    base_path = REPOSITORY / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert base["design_sha256"] == json.loads(
        base_path.read_text(encoding="ascii")
    )["design_sha256"]

    commit = design["implementation_git_commit"]
    source = design["source_binding"]
    for prefix in (
        "event_scaling",
        "event_training",
        "event_model_operator",
        "segmented_model_operator",
        "preflight_runner",
        "event_scaling_test",
        "event_training_test",
        "segmented_model_operator_test",
    ):
        assert source[f"{prefix}_sha256"] == _git_blob_sha256(
            commit,
            source[f"{prefix}_path"],
        )
    directml_path = REPOSITORY / source["directml_evidence_path"]
    assert source["directml_evidence_file_sha256"] == _file_sha256(directml_path)
    directml = json.loads(directml_path.read_text(encoding="ascii"))
    assert directml["implementation_git_commit"] == commit
    assert directml["directml_runtime"]["backend_vendor"] == "AMD Radeon RX 9070 XT"
    assert directml["directml_runtime"]["warning_count"] == 0
    assert directml["directml_runtime"]["cpu_fallback_warning_count"] == 0
    assert directml["directml_probe"]["all_candidates_trained"] is True
    assert directml["directml_probe"]["market_data_used"] is False

    delta = design["declared_delta"]
    assert delta["event_scaler_schema_version"] == (
        "round-074-event-feature-scaler-v6"
    )
    assert delta["event_training_schema_version"] == "round-074-event-training-v28"
    assert delta["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v27"
    )
    assert delta["selection_protocol_schema_version"] == (
        "round-074-event-selection-protocol-v2"
    )
    assert delta["legacy_unbound_scaler_rejected_in_segmented_mode"] is True
    assert delta["sealed_test_consumed"] is False

    training = design["training_selection_contract"]
    assert training["minimum_optimizer_run_count"] == 128
    assert training["minimum_early_stopping_run_count"] == 32
    assert training["minimum_purge_ns"] == 310500000000
    assert training["feature_value_used_for_assignment"] is False
    assert training["target_label_used_for_assignment"] is False
    assert training["model_output_used_for_assignment"] is False

    scaling = design["preprocessing_isolation_contract"]
    assert scaling["fit_source_scope"] == (
        "segmented_optimization_training_runs"
    )
    assert scaling["early_stopping_features_used_for_fit"] is False
    assert scaling["tuning_features_used_for_fit"] is False
    assert scaling["sealed_test_features_used_for_fit"] is False
    assert scaling["policy_loader_requires_exact_optimizer_run_identity"] is True
    assert scaling["transductive_covariate_leakage_permitted"] is False

    selection = design["adaptive_selection_contract"]
    assert selection["promotion_stage_order"] == [
        "architecture",
        "clock_features",
        "order_flow_features",
        "state_conditioned_flow",
        "causal_pretraining",
    ]
    assert selection["promotion_stage_scheduled_slot_bounds"] == [
        514,
        525,
        535,
        546,
        556,
        566,
    ]
    assert selection["promotion_stage_panels_are_pairwise_disjoint"] is True
    assert selection["same_promotion_panel_reused_across_stages"] is False
    assert selection["checkpoint_selection_uses_training_early_stopping_only"] is True
    assert selection["sealed_test_used_for_model_selection"] is False

    assert len(design["research_basis"]) == 4
    assert all(item["transfer_limit"] for item in design["research_basis"])
    verification = design["verification"]
    assert verification["connected_test_count_before_v92_integrity_test"] == 105
    assert verification["expected_connected_test_count_with_v92_integrity"] == 106
    assert verification["current_commit_amd_directml_preflight_passed"] is True
    assert verification["full_representative_training_performed"] is False
    assert design["evidence_boundary"]["predictive_accuracy_evaluated"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert not any(design["authority"].values())
