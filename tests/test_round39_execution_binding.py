from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/model-research/action-value"
BINDING = RESEARCH / "round-039-causal-refit-utility-ai-execution-binding.json"
DESIGN = RESEARCH / "round-039-causal-refit-utility-ai-ablation-design.json"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def test_round39_execution_binding_is_canonical_and_blob_complete() -> None:
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    canonical = dict(binding)
    claimed = canonical.pop("binding_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert binding["schema_version"] == (
        "round-039-causal-refit-utility-ai-execution-binding-v1"
    )
    assert binding["round"] == 39
    assert binding["design_sha256"] == design["design_sha256"]
    assert binding["source_certificate"]["canonical_sha256"] == (
        "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
    )
    assert binding["source_certificate"]["source_round"] == 38
    implementation_commit = binding["implementation_commit"]
    assert len(implementation_commit) == 40
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            implementation_commit,
            "HEAD",
        ],
        check=True,
    )
    blobs = {item["path"]: item["git_blob_oid"] for item in binding["blobs"]}
    required = {
        "docs/model-research/action-value/round-039-causal-refit-utility-ai-ablation-design.json",
        "src/simple_ai_trading/rolling_refit_model.py",
        "src/simple_ai_trading/rolling_refit_ai_veto.py",
        "tools/run_causal_refit_utility_ai_ablation.py",
        "tests/test_round39_execution_binding.py",
    }
    assert required <= set(blobs)
    for path, expected_oid in blobs.items():
        assert _git("rev-parse", f"{implementation_commit}:{path}") == expected_oid
        assert _git("rev-parse", f"HEAD:{path}") == expected_oid

    execution = binding["execution"]
    assert execution["candidate_count"] == 4
    assert execution["monthly_refits"] == 6
    assert execution["model_artifact_count"] == 60
    assert execution["ai_models"] == ["qwen3:8b", "fino1:8b"]
    assert execution["ai_batch_size"] == 12
    for field in (
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
    ):
        assert binding["governance"][field] is False
