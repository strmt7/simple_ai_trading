from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-segmented-v3-training-quota-impossibility-2026-08-03.json"
)


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


def test_round74_segmented_training_quota_is_provably_unreachable() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")
    snapshot = payload["snapshot"]
    training = payload["training_role"]
    scope = payload["scope"]
    upper_bound = (
        training["admitted_eligible_anchor_ns"]
        + (
            snapshot["resultless_started_training_slot_count"]
            + training["future_training_slot_count"]
        )
        * training["generous_maximum_anchor_per_unresolved_or_future_slot_ns"]
    )

    assert claimed == _canonical_sha256(payload)
    assert upper_bound == training["maximum_possible_anchor_ns"]
    assert training["deficit_to_requirement_ns"] == (
        training["required_eligible_anchor_ns"] - upper_bound
    )
    assert upper_bound < training["required_eligible_anchor_ns"]
    assert training["quota_mathematically_possible"] is False
    assert payload["decision"] == "campaign_cannot_qualify_model"
    assert scope == {
        "active_database_opened": False,
        "active_wal_opened": False,
        "market_target_data_accessed": False,
        "model_fit_performed": False,
        "orders_submitted": False,
        "profitability_or_edge_claim": False,
        "slot_directories_and_terminal_results_read": True,
        "trading_authority": False,
    }
