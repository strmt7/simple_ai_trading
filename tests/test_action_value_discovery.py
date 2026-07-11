from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_action_value_discovery import (
    discovery_candidates,
    load_discovery_design,
)


def _tracked_design(round_number: int = 9) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "action-value"
        / f"round-{round_number:03d}-design.json"
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


def test_round_ten_design_pins_calibration_fix_and_medium_horizons() -> None:
    design = load_discovery_design(_tracked_design(10))
    candidates = discovery_candidates(design)

    assert design["design_sha256"] == (
        "a2aa45f8245a12a85ea94365333f621fc2824a425ad6731105253a138fb0e049"
    )
    assert design["change_control"]["implementation_commit"] == (
        "58e6ac5f75bccb75739c6084c4861ba2ecc981fe"
    )
    assert design["training"]["calibration_method"] == (
        "base_rate_initialized_damped_platt_v2"
    )
    assert len(candidates) == 12
    assert candidates[0]["candidate_id"] == "conservative-h300"
    assert candidates[-1]["candidate_id"] == "aggressive-h1800"
    assert all(
        candidate["horizon_seconds"] in {300, 600, 900, 1800}
        for candidate in candidates
    )
