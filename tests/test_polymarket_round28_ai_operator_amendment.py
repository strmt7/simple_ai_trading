from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "docs/model-research/polymarket/"
    "round-028-ai-operator-implementation-amendment-v1.json"
)
EXPECTED_SHA256 = "41f92d0625ee9fb4c53a4d66c8decdb3f5c053c44ce146316778e7431ba2b04d"


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


def test_round28_ai_operator_amendment_binds_exact_sources() -> None:
    artifact = json.loads(AMENDMENT.read_text(encoding="ascii"))
    claimed = artifact.pop("amendment_sha256")

    assert claimed == EXPECTED_SHA256
    assert _canonical_sha256(artifact) == claimed
    assert artifact["knowledge_at_freeze"]["official_outcomes_accessed_by_ai_operator"] is False
    assert artifact["test_scope"]["financial_result"] is False
    assert artifact["limitations"]
    assert set(artifact["authority"].values()) == {False}
    for relative, expected in artifact["source_text_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
