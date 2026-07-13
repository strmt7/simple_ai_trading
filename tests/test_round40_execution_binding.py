from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BINDING = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-040-causal-meta-label-execution-binding.json"
)
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-040-causal-meta-label-capacity-ai-design.json"
)


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


def test_round40_binding_is_hash_bound_to_exact_implementation() -> None:
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    canonical = dict(binding)
    claimed = canonical.pop("binding_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "6ba64e3bd6045f8f96d7b69758de1388929a1aee8706231ae12c1a9ee17cdb9f"
    assert binding["schema_version"] == (
        "round-040-causal-meta-label-execution-binding-v1"
    )
    assert binding["round"] == 40
    assert binding["design_sha256"] == (
        "9e48a88e183f4978b169e011ef300a3f77488cf5e5961d2dec82d549c46300ff"
    )
    assert binding["implementation_commit"] == (
        "b7e5852ed0b8fae3a02204474d7ae0831bc9bef5"
    )
    assert binding["source_certificate"]["canonical_sha256"] == (
        "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
    )
    assert binding["source_certificate"]["file_sha256"] == (
        "e2fe434d7c290f09913160506c52fce30849a6bd319465390c4b4d22dad482a7"
    )
    execution = binding["execution"]
    assert execution["candidate_count"] == 1
    assert execution["primary_model_artifact_count"] == 18
    assert execution["meta_model_artifact_count"] == 6
    assert execution["threshold_cells"] == 216
    assert execution["maximum_entries_per_symbol_day"] == 8
    assert execution["ai_model"] == "DianJin/DianJin-R1-7B"
    assert execution["ai_maximum_cases"] == 180
    assert binding["governance"]["development_only"] is True
    assert binding["governance"]["selection_contaminated"] is True
    for field in (
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
    ):
        assert binding["governance"][field] is False

    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    assert design["design_sha256"] == binding["design_sha256"]
    implementation = binding["implementation_commit"]
    for artifact in binding["blobs"]:
        assert _git("rev-parse", f"{implementation}:{artifact['path']}") == (
            artifact["git_blob_oid"]
        )
        assert _git("rev-parse", f"HEAD:{artifact['path']}") == (
            artifact["git_blob_oid"]
        )
