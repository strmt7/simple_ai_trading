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
    / "round-074-local-ai-review-design-v69.json"
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


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_v69_is_hash_and_source_bound() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design_binding"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == hashlib.sha256(base_path.read_bytes()).hexdigest()
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )
    implementation_commit = design["implementation_git_commit"]
    for label, source in design["source_binding"].items():
        if label == "file_sha256_normalization":
            continue
        assert source["sha256"] == _git_blob_sha256(
            implementation_commit,
            source["path"],
        )


def test_round74_v69_corrects_ai_financial_semantics_without_claiming_edge() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    audit = design["model_research_audit"]
    host = design["host_observation"]
    evidence = design["evidence_boundary"]

    assert delta["payoff_forecast_quantity"] == "capital_scaled_net_payoff_bps"
    assert delta["payoff_forecast_is_gross_return"] is False
    assert delta["ai_may_deduct_execution_costs_again"] is False
    assert delta["regime_unpredictability_is_binary_event_probability"] is False
    assert delta["background_knowledge_may_fill_missing_premises"] is False
    assert delta["model_panel_changed"] is False
    assert audit["fin_r1"]["upstream_license_declared"] is False
    assert audit["fin_r1"]["operational_candidate"] is False
    assert audit["agentar_fin_r1"]["operational_candidate"] is False
    assert audit["oda_fin_rl_8b"]["operational_candidate"] is False
    assert host["free_physical_memory_gib"] < host["minimum_required_free_memory_gib"]
    assert host["live_ai_preflight_executed"] is False
    assert evidence["prompt_semantics_match_round74_training_targets"] is True
    assert evidence["model_behavior_improved_established"] is False
    assert evidence["ai_improves_ml_established"] is False
    assert evidence["profitability_established"] is False
    assert set(design["authority"].values()) == {False}
