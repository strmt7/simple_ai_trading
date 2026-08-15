from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "docs/model-research/polymarket/"
    "round-028-ai-sealed-evaluation-implementation-amendment-v1.json"
)
EXPECTED_SHA256 = "925fc1274dd199648672bcab6f33013ec51a97a239cb2ccac0580dbf167a137d"


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


def test_round28_ai_sealed_amendment_binds_exact_sources() -> None:
    artifact = json.loads(AMENDMENT.read_text(encoding="ascii"))
    claimed = artifact.pop("amendment_sha256")

    assert claimed == EXPECTED_SHA256
    assert _canonical_sha256(artifact) == claimed
    assert artifact["knowledge_at_freeze"][
        "official_outcomes_accessed_by_ai_operator"
    ] is False
    assert artifact["test_scope"]["financial_result"] is False
    assert artifact["test_scope"]["terminal_restart_target_store_opened"] is False
    assert artifact["limitations"]
    assert set(artifact["authority"].values()) == {False}
    for relative, expected in artifact["source_text_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
