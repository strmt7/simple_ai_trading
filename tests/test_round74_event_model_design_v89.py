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
    / "round-074-event-sequence-model-design-v89.json"
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


def test_round74_v89_binds_grouped_equal_run_pretraining() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v89"

    base = design["base_design"]
    base_path = REPOSITORY / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert base["design_sha256"] == json.loads(
        base_path.read_text(encoding="ascii")
    )["design_sha256"]

    source = design["source_binding"]
    commit = design["implementation_git_commit"]
    for section in ("model", "pretraining", "training", "tests"):
        assert source[section]["sha256"] == _git_blob_sha256(
            commit,
            source[section]["path"],
        )
    assert source["model"]["model_schema_version"] == (
        "round-074-event-payoff-model-v7"
    )
    assert source["pretraining"]["schema_version"] == (
        "round-074-causal-next-event-pretraining-v2"
    )
    assert source["training"]["training_schema_version"] == (
        "round-074-event-training-v24"
    )
    assert source["training"]["pretest_policy_schema_version"] == (
        "round-074-event-pretest-policy-v23"
    )

    split = design["split_reuse_contract"]
    assert split["split_construction_count_per_pretrained_candidate_fit"] == 1
    assert split["split_hash_and_feature_hash_construction_repeated_per_seed"] is False
    assert split["target_only_batch_replacement_permitted"] is True
    assert split["copied_or_substituted_feature_tensor_rejected"] is True
    assert split["policy_reload_requires_one_shared_split_across_pretrained_peers"]

    batching = design["device_batching_contract"]
    assert batching["default_capture_runs_per_device_group"] == 8
    assert batching["one_separate_loss_mean_per_capture_run"] is True
    assert batching["optimizer_step_count_changed"] is False
    assert batching["validation_weighting_changed"] is False

    directml = design["verification"]["directml_evidence"]
    assert directml["file_sha256"] == _file_sha256(
        REPOSITORY / directml["path"]
    )
    assert design["evidence_boundary"][
        "pretraining_improves_predictive_accuracy_claim"
    ] is False
    assert design["authority"]["financial_edge_tested"] is False
