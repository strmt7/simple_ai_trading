from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BINDING = (
    ROOT
    / "docs/model-research/action-value"
    / "round-041-prequential-meta-label-execution-binding.json"
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


def test_round41_binding_is_hash_bound_to_exact_prequential_stack() -> None:
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    canonical = dict(binding)
    claimed = canonical.pop("binding_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "3de2130506ea026c9834489f51ce236ad389df793133fd1e67433e5ca2bcb50e"
    assert binding["schema_version"] == (
        "round-041-prequential-meta-label-execution-binding-v1"
    )
    assert binding["round"] == 41
    assert binding["design_sha256"] == (
        "367aa94f4ea435ac4a14509c698c1de986d7ebd9adc097046da1565a5708cd2e"
    )
    assert binding["implementation_commit"] == (
        "ebf8d98263d67de20f52096adfaa0c2e8d1f2c50"
    )
    execution = binding["execution"]
    assert execution["primary_target_months"] == 14
    assert execution["primary_model_artifact_count"] == 42
    assert execution["meta_evaluation_months"] == 6
    assert execution["meta_model_artifact_count"] == 6
    assert execution["threshold_cells"] == 216
    assert execution["ai_model"] == "DianJin/DianJin-R1-7B"
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

    implementation = binding["implementation_commit"]
    for artifact in binding["blobs"]:
        expected = artifact["git_blob_oid"]
        assert _git("rev-parse", f"{implementation}:{artifact['path']}") == expected
        assert _git("rev-parse", f"HEAD:{artifact['path']}") == expected
