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
    / "round-074-event-sequence-model-design-v69.json"
)
PREVIOUS = DESIGN.with_name("round-074-event-sequence-model-design-v68.json")
IMPLEMENTATION_COMMIT = "87f6bac42201c02cd465ce53a5f0097da39fbdba"


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


def _normalized_lf_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def _git_file_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v69_binds_reloadable_scaler_and_latency_component() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    scaler = design["pretest_scaler_artifact"]
    latency = design["online_decision_component_latency"]

    assert claimed == _canonical_sha256(design)
    assert design["base_design"]["normalized_lf_sha256"] == (
        _normalized_lf_sha256(PREVIOUS)
    )
    assert (
        design["base_design"]["design_sha256"]
        == json.loads(PREVIOUS.read_text(encoding="ascii"))["design_sha256"]
    )
    assert scaler["training_source_normalized_lf_sha256"] == (
        _git_file_sha256(IMPLEMENTATION_COMMIT, scaler["training_source_path"])
    )
    assert latency["source_normalized_lf_sha256"] == _git_file_sha256(
        IMPLEMENTATION_COMMIT, latency["source_path"]
    )
    assert latency["test_normalized_lf_sha256"] == _git_file_sha256(
        IMPLEMENTATION_COMMIT, latency["test_path"]
    )
    assert scaler["cohort_mode_requires_exact_fitted_scaler"] is True
    assert scaler["reload_equivalence_verified_before_policy_publication"] is True
    assert latency["measurements_per_profile"] == 300
    assert latency["tail_quantile"] == 0.99
    assert latency["tail_confidence"] == 0.95


def test_round74_v69_forbids_partial_latency_from_becoming_execution_evidence() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    rule = design["latency_composition_and_replay_rule"]

    assert rule["component_may_be_called_end_to_end_tick_to_trade"] is False
    assert len(rule["missing_required_components"]) == 4
    assert (
        rule["policy_selection_may_use_precomputed_baseline_payoff_after_component_measurement"]
        is False
    )
    assert rule["threshold_specific_database_replay_permitted"] is False
    assert rule["candidate_independent_raw_replay_panel_required"] is True
    assert (
        rule["testnet_execution_calibration_may_establish_mainnet_equivalence"]
        is False
    )


def test_round74_v69_preserves_all_authority_blocks() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))

    assert set(design["authority"].values()) == {False}
    assert design["verification"]["host_gpu_benchmark_run"] is False
    assert design["verification"]["prospective_model_result_exists"] is False
    assert design["unchanged_model_contract"]["conservative_default"] is True
    assert design["unchanged_model_contract"]["test_role_accessed"] is False
