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
    / "round-074-local-ai-review-design-v72.json"
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


def test_round74_ai_design_v72_binds_epistemic_disagreement_prompt() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    disagreement = artifact["disagreement_contract"]
    missing = artifact["missing_diagnostics_contract"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == "round-074-local-ai-review-design-v72"
    assert artifact["supersedes_artifact_sha256"] == (
        "538d1b98a893060950a37c453dc10ecfe244ef1509d83bfc6c8709df6f5a67e5"
    )
    for binding in source["implementation_files"] + source["contract_tests"]:
        assert binding["sha256"] == _source_file_sha256_at(
            commit,
            binding["path"],
        )
    predecessor = source["predecessor_design"]
    assert predecessor["sha256"] == hashlib.sha256(
        (REPOSITORY / predecessor["path"]).read_bytes()
    ).hexdigest()

    assert artifact["schema_contract"] == {
        "review_request_schema_version": "round-074-ai-review-request-v7",
        "prompt_payload_schema_version": "round-074-ai-prompt-payload-v10",
        "bridge_schema_version": "round-074-ai-bridge-v7",
        "review_panel_schema_version": "round-074-ai-review-panel-v17",
        "contract_screen_schema_version": "round-074-ai-contract-screen-v3",
        "contract_case_schema_version": "round-074-ai-contract-case-v3",
        "runtime_preflight_schema_version": "round-074-local-ai-runtime-preflight-v9",
    }
    assert disagreement["standard_deviation_definition"] == "population"
    assert disagreement["payoff_quantile_summary"].startswith("root_mean_square")
    assert disagreement["interpreted_as_calibrated_confidence"] is False
    assert disagreement["interpreted_as_predictive_edge"] is False
    assert disagreement["may_only_support_veto_or_risk_reduction"] is True
    assert disagreement["may_create_or_reverse_trade_side"] is False
    assert disagreement["may_increase_size_risk_or_leverage"] is False
    assert missing == {
        "peer_count": 0,
        "all_disagreement_values": 0.0,
        "available": False,
        "zero_values_mean_confident_agreement": False,
        "missing_values_imputed": False,
        "missing_values_fabricated": False,
    }
    assert artifact["production_panel"]["model_names"] == [
        "fino1:8b",
        "qwen3:8b",
    ]
    assert artifact["research_challenger_panel"]["model_names"] == [
        "fin-r1:8b",
        "oda-fin-rl:8b",
    ]
    assert artifact["verification"]["focused_tests_passed_after_strict_parser_hardening"] == 56
    assert artifact["verification"]["connected_tests_passed_after_prompt_and_schema_change"] == 110
    assert artifact["verification"]["representative_market_ai_evaluation_performed"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
