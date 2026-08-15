from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "docs/model-research/polymarket/"
    "round-028-ai-sealed-evaluation-implementation-amendment-v2.json"
)
EXPECTED_SHA256 = "84b6d4bcf767f12c53bef87ddbf8458606cb24487ced0f8a703a6dcf1f6d2681"


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


def test_round28_ai_sealed_v2_amendment_binds_safe_process_sources() -> None:
    artifact = json.loads(AMENDMENT.read_text(encoding="ascii"))
    claimed = artifact.pop("amendment_sha256")

    assert claimed == EXPECTED_SHA256
    assert _canonical_sha256(artifact) == claimed
    assert artifact["correction"]["authoritative_operator"].endswith(
        "_enveloped.py"
    )
    assert artifact["side_effect_policy"] == {
        "historical_cutoff_changed": False,
        "prospective_capture_changed": False,
        "prospective_service_stopped_or_restarted": False,
        "reusable_historical_data_opened_or_rewritten": False,
        "repository_historical_cutoff_exclusive_utc": "2026-08-14T00:00:00Z",
    }
    assert artifact["test_scope"]["financial_result"] is False
    assert set(artifact["authority"].values()) == {False}
    for relative, expected in artifact["source_text_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
