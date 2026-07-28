from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_CLOCK_FEATURE_NAMES,
    ROUND74_EVENT_CLOCK_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_FEATURE_VIEWS,
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v87.json"
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
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_v87_binds_mandatory_exchange_clock_ablation() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v87"

    source = design["source_binding"]
    commit = design["implementation_git_commit"]
    assert source["training"]["sha256"] == _git_blob_sha256(
        commit,
        source["training"]["path"],
    )
    assert source["tests"]["sha256"] == _git_blob_sha256(
        commit,
        source["tests"]["path"],
    )
    assert source["training"]["training_schema_version"] == (
        ROUND74_EVENT_TRAINING_SCHEMA_VERSION
    )
    assert source["training"]["pretest_policy_schema_version"] == (
        ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
    )
    assert source["event_feature_contract"][
        "exchange_clock_feature_names_sha256"
    ] == ROUND74_EVENT_CLOCK_FEATURE_NAMES_SHA256

    feature_view = design["feature_view_contract"]
    assert feature_view["ordered_views"] == list(ROUND74_EVENT_FEATURE_VIEWS)
    assert feature_view["clock_neutral_masked_features"] == list(
        ROUND74_EVENT_CLOCK_FEATURE_NAMES
    )
    assert feature_view["training_calibration_sealed_test_and_inference_parity"] is True

    stage_2 = design["selection_contract"]["stage_2"]
    assert stage_2["incomplete_panel_outcome"] == "clock_neutral"
    assert stage_2["gate_failure_outcome"] == "clock_neutral"
    assert stage_2["promotion_requires_every_paired_run_noninferior"] is True
    assert (
        stage_2["promotion_requires_every_run_symbol_horizon_subgroup_noninferior"]
        is True
    )
    assert design["selection_contract"][
        "recorded_winner_trusted_without_recomputation"
    ] is False
    assert design["evidence_boundary"]["clock_features_selected_on_real_data"] is False
    assert design["authority"]["financial_edge_tested"] is False
