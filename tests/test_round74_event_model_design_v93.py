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
    / "round-074-event-sequence-model-design-v93.json"
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


def test_round74_v93_persists_complete_training_split_provenance() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v93"

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
        "development_operator",
        "segmented_model_operator",
        "event_training_test",
        "segmented_model_operator_test",
    ):
        assert source[f"{prefix}_sha256"] == _git_blob_sha256(
            commit,
            source[f"{prefix}_path"],
        )
    evidence_path = REPOSITORY / source["directml_evidence_path"]
    assert source["directml_evidence_file_sha256"] == _file_sha256(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    assert evidence["implementation_git_commit"] == commit
    assert evidence["directml_runtime"]["backend_vendor"] == "AMD Radeon RX 9070 XT"
    assert evidence["directml_runtime"]["warning_count"] == 0
    assert evidence["directml_runtime"]["cpu_fallback_warning_count"] == 0
    assert evidence["directml_probe"]["all_candidates_trained"] is True
    assert evidence["directml_probe"]["market_data_used"] is False

    delta = design["declared_delta"]
    assert delta["event_training_schema_version"] == "round-074-event-training-v29"
    assert delta["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v28"
    )
    assert delta["selection_protocol_schema_version"] == (
        "round-074-event-selection-protocol-v3"
    )
    assert delta["training_split_schema_version"] == (
        "round-074-segmented-training-selection-split-v1"
    )
    assert delta["sealed_test_consumed"] is False

    completeness = design["artifact_completeness_contract"]
    assert all(completeness.values())
    binding = design["cross_artifact_binding_contract"]
    assert all(binding.values())
    research = design["research_inheritance"]
    assert research["base_v92_primary_source_review_inherited"] is True
    assert research["new_predictive_or_financial_hypothesis_introduced"] is False
    assert research["prior_transfer_limits_remain_in_force"] is True

    verification = design["verification"]
    assert verification["connected_test_count_before_v93_integrity_test"] == 106
    assert verification["expected_connected_test_count_with_v93_integrity"] == 107
    assert verification["current_commit_amd_directml_preflight_passed"] is True
    assert verification["current_commit_directml_cpu_fallback_warning_count"] == 0
    assert verification["full_representative_training_performed"] is False
    assert design["evidence_boundary"]["predictive_accuracy_evaluated"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert not any(design["authority"].values())
