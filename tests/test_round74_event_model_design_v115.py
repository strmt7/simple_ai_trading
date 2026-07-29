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
    / "round-074-event-sequence-model-design-v115.json"
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


def test_round74_v115_binds_coherent_payoff_head() -> None:
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


def test_round74_v115_blocks_impossible_probability_without_edge_claim() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    constraint = design["financial_constraint"]
    verification = design["verification"]

    assert delta["event_model_schema_version"] == "round-074-event-payoff-model-v11"
    assert delta["new_probability_head"] == (
        "three_outcome_softmax_equivalent_exposed_as_two_marginal_logits"
    )
    assert delta["positive_probability_sum_bounded_by_one"] is True
    assert delta["directional_competition_has_cross_side_gradient"] is True
    assert delta["common_positive_temperature_preserves_probability_simplex"] is True
    assert delta["parameter_counts_changed"] is False
    assert delta["forward_passes_added"] == 0
    assert delta["backward_passes_added"] == 0
    assert constraint["jointly_eligible_long_and_short_can_both_be_positive"] is False
    assert verification["unique_tests_passed"] == 181
    assert verification["representative_training_performed"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["directml_preflight"]["cpu_fallback_warning_count"] == 0
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["drawdown_claim"] is False
    assert set(design["authority"].values()) == {False}
