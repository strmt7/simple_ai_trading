from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round25_forensic_partition import (
    validate_round25_forensic_partition_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PARTITION = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-v2-forensic-partition-2026-08-14.json"
)
FILE_SHA256 = "d28605f00e01539298cf1efa92b3bbea7abe2e32467a7cfd3fb4131b69f98e99"
PARTITION_SHA256 = "eb2fcc3e3db60d5b5979ddf129a83c6b57a831e3c3a0cd8e4d2bbcf8729b167d"
FEATURE_STORE_MANIFEST_SHA256 = (
    "838c4c2e3a2b59e8955ae97ec76a532e53a3b1b849d11275230c1c604854268f"
)
VENUE_PARAMETER_AUDIT_SHA256 = (
    "be5cb9626d8fff531cb0d4e3d9feac520e6e82e1019476d4486e3c8950b5fa67"
)


def test_forensic_partition_is_target_blind_self_hashed_and_source_bound() -> None:
    assert hashlib.sha256(PARTITION.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(PARTITION.read_text(encoding="ascii"))

    validated = validate_round25_forensic_partition_manifest(value)

    assert validated["partition_sha256"] == PARTITION_SHA256
    assert validated["feature_store_manifest_sha256"] == (
        FEATURE_STORE_MANIFEST_SHA256
    )
    assert validated["venue_parameter_audit_sha256"] == (
        VENUE_PARAMETER_AUDIT_SHA256
    )
    assert validated["role_counts"] == {
        "calibration": 12,
        "purged": 4,
        "selection": 14,
        "train": 42,
    }
    assert validated["outcomes_consulted"] is False
    assert validated["model_scores_consulted"] is False
    assert validated["selection_accessed"] is False
    assert validated["profitability_claim"] is False


def test_forensic_partition_rejects_role_tampering() -> None:
    value = json.loads(PARTITION.read_text(encoding="ascii"))
    value["conditions"][0]["role"] = "selection"

    with pytest.raises(ValueError, match="partition"):
        validate_round25_forensic_partition_manifest(value)
