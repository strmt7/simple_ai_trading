from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "complete-set-midpoint-uplift-contract-v1.json"
)
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "complete-set-midpoint-uplift-v1-2026-08-26.json"
)
TOOL_PATH = ROOT / "tools" / "screen_polymarket_complete_set_midpoint_uplift.py"
EXPECTED_CONTRACT_HASH = (
    "5ae9c7ad1a7a3e7f68f546b76501bf72bf1558ba3d9a570efd1050798d08b5ad"
)
EXPECTED_ARTIFACT_HASH = (
    "33cdc53555f8bbdecf6a9977a77d2c3bc004dab4bff27abb36eac4452f96e5a3"
)
EXPECTED_TOOL_HASH = "c6066330e17b89f480c5939c79dfbffee641b6446ce76ac9812c7815510a7b6f"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _embedded_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_midpoint_uplift_evidence_is_source_bound() -> None:
    contract = _load(CONTRACT_PATH)
    artifact = _load(ARTIFACT_PATH)

    assert contract["contract_sha256"] == EXPECTED_CONTRACT_HASH
    assert _embedded_hash(contract, "contract_sha256") == EXPECTED_CONTRACT_HASH
    assert artifact["result_sha256"] == EXPECTED_ARTIFACT_HASH
    assert _embedded_hash(artifact, "result_sha256") == EXPECTED_ARTIFACT_HASH
    assert artifact["implementation"] == {
        "path": "tools/screen_polymarket_complete_set_midpoint_uplift.py",
        "sha256": EXPECTED_TOOL_HASH,
    }
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH


def test_all_eligible_complete_sets_have_unit_midpoint_value() -> None:
    artifact = _load(ARTIFACT_PATH)
    summary = artifact["summary"]

    assert summary["eligible_market_count"] == 55
    assert summary["complete_midpoint_market_count"] == 55
    assert summary["maximum_midpoint_sum"] == "1.0000"
    assert summary["public_uplift_candidate_count"] == 0
    assert summary["history_or_book_escalation_permitted"] is False
    assert {Decimal(str(row["midpoint_sum"])) for row in artifact["markets"]} == {
        Decimal("1")
    }
    assert all(row["public_uplift_candidate"] is False for row in artifact["markets"])
