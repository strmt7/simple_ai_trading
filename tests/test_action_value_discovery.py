from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_action_value_discovery import (
    discovery_candidates,
    load_discovery_design,
)


def _tracked_design() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "action-value"
        / "round-009-design.json"
    )


def test_round_nine_design_is_hash_bound_and_expands_to_twelve_candidates() -> None:
    design = load_discovery_design(_tracked_design())
    candidates = discovery_candidates(design)

    assert design["design_sha256"] == (
        "a6ac6be9d4322f1b78a5894c72e131b5ef596712dfd2decaff32c969373e76e6"
    )
    assert len(candidates) == 12
    assert candidates[0]["candidate_id"] == "conservative-h60"
    assert candidates[-1]["candidate_id"] == "aggressive-h900"
    assert all(candidate["horizon_seconds"] in {60, 120, 300, 900} for candidate in candidates)


def test_round_nine_design_rejects_post_commit_mutation(tmp_path) -> None:
    payload = json.loads(_tracked_design().read_text(encoding="utf-8"))
    payload["execution"]["taker_fee_bps_per_side"] = 0.0
    changed = tmp_path / "changed-design.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_discovery_design(changed)
