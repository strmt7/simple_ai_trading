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
    / "round-074-event-sequence-model-design-v114.json"
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


def test_round74_v114_binds_path_risk_rearrangement() -> None:
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


def test_round74_v114_preserves_gradients_without_financial_claim() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    preflight = design["directml_preflight"]
    verification = design["verification"]

    assert delta["event_model_schema_version"] == ("round-074-event-payoff-model-v10")
    assert delta["previous_projection"] == "one_sided_cumulative_maximum"
    assert delta["new_projection"] == (
        "elementwise_monotone_rearrangement_by_minimum_maximum_insertion_network"
    )
    assert delta["raw_horizon_surface_multiset_preserved"] is True
    assert delta["raw_horizon_surface_may_be_discarded"] is False
    assert delta["output_horizon_order_exact"] is True
    assert delta["output_quantile_order_exact"] is True
    assert delta["parameter_counts_changed"] is False
    assert delta["training_objective_changed"] is False
    assert delta["forward_passes_added"] == 0
    assert delta["backward_passes_added"] == 0
    assert preflight["backend_kind"] == "directml"
    assert preflight["backend_vendor"] == "AMD Radeon RX 9070 XT"
    assert preflight["backend_accelerated"] is True
    assert preflight["cpu_fallback_warning_count"] == 0
    assert preflight["financial_edge_tested"] is False
    assert verification["unique_tests_passed"] == 130
    assert (
        verification["each_rearranged_horizon_has_exactly_one_live_raw_gradient_source"]
        is True
    )
    assert verification["representative_training_performed"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["drawdown_claim"] is False
    assert set(design["authority"].values()) == {False}
