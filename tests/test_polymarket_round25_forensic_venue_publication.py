from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round25_forensic_venue import (
    POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256,
    validate_round25_forensic_venue_parameter_audit,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-v2-forensic-venue-parameters-2026-08-14.json"
)


def test_forensic_venue_parameters_are_target_blind_and_self_hashed() -> None:
    value = json.loads(ARTIFACT.read_text(encoding="ascii"))

    validated = validate_round25_forensic_venue_parameter_audit(value)

    assert validated["artifact_sha256"] == (
        POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256
    )
    assert validated["query_boundary"]["outcomes_or_resolutions_read"] is False
    assert validated["economic_replay_binding"]["all_captured_conditions_match"] is True

    tampered = json.loads(json.dumps(value))
    tampered["observations"]["tick_size_groups"][0]["tick_size"] = "0.001"
    with pytest.raises(ValueError, match="audit differs"):
        validate_round25_forensic_venue_parameter_audit(tampered)
