from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-local-ai-review-design-v71.json"
)


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_file_sha256_at(commit: str, path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{path}"],  # nosec B607
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_ai_design_v71_binds_oda_research_harness() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    challengers = artifact["research_challenger_panel"]
    contract = artifact["research_harness_contract"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == "round-074-local-ai-review-design-v71"
    assert artifact["supersedes_artifact_sha256"] == (
        "b90804ba140acef5fe0822898f3af7e4e53c55034d3bfe0f47d50df640066be0"
    )
    for label in ("ai_review_preparation", "ai_contract_screen_test"):
        binding = source[label]
        assert binding["sha256"] == _source_file_sha256_at(
            commit,
            binding["path"],
        )
    predecessor = source["predecessor_design"]
    assert predecessor["sha256"] == hashlib.sha256(
        (REPOSITORY / predecessor["path"]).read_bytes()
    ).hexdigest()

    assert artifact["production_panel"]["model_names"] == [
        "fino1:8b",
        "qwen3:8b",
    ]
    assert artifact["production_panel"]["oda_included"] is False
    assert [value["runtime_model_name"] for value in challengers] == [
        "fin-r1:8b",
        "oda-fin-rl:8b",
    ]
    oda = challengers[1]
    assert oda["model_id"] == "OpenDataArena/ODA-Fin-RL-8B"
    assert oda["parameter_count"] == 8_190_735_360
    assert oda["model_artifact_kind"] == "gguf"
    assert oda["quantization"] == "q4_k_m"
    assert oda["model_artifact_sha256"] == (
        "d40d1dd4105be8d85cbb444cb58e92c4882623f0baa4dea5d296745d6bc13861"
    )
    assert oda["upstream_model_revision_is_artifact_publisher_revision"] is False
    assert contract["challengers_are_disjoint_from_production_manifests"] is True
    assert contract["implicit_qualification_permitted"] is False
    assert contract["implicit_default_selection_permitted"] is False
    assert contract["adversarial_contract_screen_required"] is True
    assert contract["paired_after_cost_ai_vs_ml_uplift_required"] is True
    assert contract["language_model_may_increase_risk_or_leverage"] is False
    assert contract["language_model_role"] == "veto_or_reduce_only"
    assert artifact["host_observation"]["oda_download_started"] is False
    assert artifact["host_observation"]["oda_model_loaded"] is False
    assert artifact["verification"]["representative_market_ai_evaluation_performed"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
