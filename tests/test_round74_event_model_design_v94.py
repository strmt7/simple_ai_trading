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
    / "round-074-event-sequence-model-design-v94.json"
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


def test_round74_v94_requires_broad_single_run_deletion_stable_promotion() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v94"

    base = design["base_design"]
    base_path = REPOSITORY / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )

    commit = design["implementation_git_commit"]
    source = design["source_binding"]
    for prefix in (
        "event_features",
        "event_training",
        "representation_comparison",
        "event_training_test",
        "representation_comparison_test",
    ):
        assert source[f"{prefix}_sha256"] == _git_blob_sha256(
            commit,
            source[f"{prefix}_path"],
        )

    evidence_path = REPOSITORY / source["directml_evidence_path"]
    assert source["directml_evidence_file_sha256"] == _file_sha256(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    assert evidence["implementation_git_commit"] == commit
    subprocess.run(  # nosec B603
        [
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            evidence["execution_git_commit"],
        ],
        cwd=REPOSITORY,
        check=True,
    )
    runtime = evidence["directml_runtime"]
    assert runtime["backend_vendor"] == "AMD Radeon RX 9070 XT"
    assert runtime["accelerated"] is True
    assert runtime["warning_count"] == 0
    assert runtime["cpu_fallback_warning_count"] == 0
    probe = evidence["directml_probe"]
    assert probe["all_candidates_trained"] is True
    assert probe["all_promotion_reports_complete"] is False
    assert probe["all_promotion_reports_promoted"] is False
    assert probe["market_data_used"] is False

    delta = design["declared_delta"]
    assert delta["event_training_schema_version"] == "round-074-event-training-v30"
    assert delta["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v29"
    )
    assert delta["selection_protocol_schema_version"] == (
        "round-074-event-selection-protocol-v4"
    )
    assert delta["feature_view_schema_version"] == "round-074-feature-view-v3"
    assert delta["representation_comparison_schema_version"] == (
        "round-074-representation-comparison-v3"
    )
    assert delta["sealed_test_consumed"] is False

    contract = design["promotion_contract"]
    assert contract["material_win_on_strict_capture_run_majority_required"] is True
    assert (
        contract[
            "every_single_capture_run_deletion_mean_above_numerical_floor_required"
        ]
        is True
    )
    assert contract["every_paired_capture_run_noninferior_required"] is True
    assert contract["every_run_symbol_horizon_subgroup_noninferior_required"] is True
    assert contract["one_capture_run_panel_can_promote"] is False
    assert contract["statistical_independence_or_significance_claim"] is False

    research = design["research_basis"]
    assert len(research["primary_sources"]) == 4
    assert research["sealed_test_remains_unmodified"] is True
    assert research["new_predictive_or_financial_hypothesis_introduced"] is False
    verification = design["verification"]
    assert verification["connected_test_count_before_v94_integrity"] == 104
    assert verification["connected_test_count_with_v94_integrity"] == 105
    assert verification["amd_directml_preflight_passed"] is True
    assert verification["directml_cpu_fallback_warning_count"] == 0
    assert verification["full_representative_training_performed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert not any(design["authority"].values())
