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
    / "round-074-event-sequence-model-design-v107.json"
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


def _source_sha256_at(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v107_binds_path_risk_coherence_implementation() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )
    implementation_commit = design["implementation_git_commit"]
    source_binding = design["source_binding"]
    assert source_binding["file_sha256_normalization"] == (
        "text_bytes_crlf_and_cr_normalized_to_lf_before_sha256"
    )
    for label, source in source_binding.items():
        if label == "file_sha256_normalization":
            continue
        assert source["sha256"] == _source_sha256_at(
            implementation_commit,
            source["path"],
        )


def test_round74_v107_enforces_path_law_without_financial_claims() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    schemas = design["schema_contract"]
    verification = design["verification"]

    assert delta["ordered_payoff_horizons_seconds"] == [1, 5, 30, 300]
    assert delta["model_mae_quantiles_nondecreasing_across_horizons"] is True
    assert delta["horizon_projection_additional_parameters"] == 0
    assert delta["horizon_projection_directml_forward_backward_supported"] is True
    assert delta["payoff_quantiles_forced_monotone_across_horizons"] is False
    assert delta["fixed_auxiliary_loss_weights_changed"] is False
    assert delta["dominated_raw_horizon_head_gradient_effect_evaluated"] is False
    assert schemas == {
        "event_dataset": "round-074-event-dataset-v11",
        "event_model": "round-074-event-payoff-model-v9",
        "event_training": "round-074-event-training-v32",
        "event_pretest_policy": "round-074-event-pretest-policy-v31",
        "event_training_preflight_run": "round-074-event-training-preflight-run-v9",
        "event_training_directml_preflight_evidence": (
            "round-074-event-training-directml-preflight-evidence-v21"
        ),
    }
    assert verification["focused_tests_passed"] == 160
    assert verification["directml_candidate_count"] == 4
    assert verification["directml_cpu_fallback_warning_count"] == 0
    assert verification["representative_training_performed"] is False
    assert design["evidence_boundary"]["path_risk_coherence_proven"] is True
    assert design["evidence_boundary"]["path_risk_accuracy_evaluated"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert set(design["authority"].values()) == {False}
