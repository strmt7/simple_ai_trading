from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_pretraining import (
    ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v88.json"
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


def test_round74_v88_binds_paired_causal_pretraining_gate() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v88"

    source = design["source_binding"]
    commit = design["implementation_git_commit"]
    for section in ("model", "pretraining", "training", "tests"):
        assert source[section]["sha256"] == _git_blob_sha256(
            commit,
            source[section]["path"],
        )
    assert source["model"]["model_schema_version"] == (
        ROUND74_EVENT_MODEL_SCHEMA_VERSION
    )
    assert source["pretraining"]["schema_version"] == (
        ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION
    )
    assert source["training"]["training_schema_version"] == (
        ROUND74_EVENT_TRAINING_SCHEMA_VERSION
    )
    assert source["training"]["pretest_policy_schema_version"] == (
        ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
    )

    pretraining = design["pretraining_contract"]
    assert pretraining["validation_boundary"].startswith("one global chronological")
    assert pretraining["per_symbol_purge_anchors"] == 128
    assert pretraining["target_mutation_changes_pretraining_split_identity"] is False
    assert pretraining["payoff_targets_read_by_pretraining"] is False
    assert pretraining["test_data_read_by_pretraining"] is False

    stage_3 = design["selection_contract"]["stage_3"]
    assert stage_3["incumbent"] == "random"
    assert stage_3["incomplete_panel_outcome"] == "random"
    assert stage_3["gate_failure_outcome"] == "random"
    assert stage_3["promotion_requires_every_paired_run_noninferior"] is True
    assert stage_3[
        "promotion_requires_every_run_symbol_horizon_subgroup_noninferior"
    ] is True
    assert design["evidence_boundary"][
        "pretraining_improves_predictive_accuracy_claim"
    ] is False
    assert design["authority"]["financial_edge_tested"] is False
