from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.publish_action_value_discovery import (
    _compact_candidate_label,
    _economics_svg,
    _forecast_svg,
)
from tools.run_action_value_discovery import (
    _canonical_sha256,
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

    with pytest.raises(ValueError, match="current design and model schemas"):
        load_discovery_design(_tracked_design(10), require_current=True)


def test_round_eleven_design_binds_current_code_registry_and_split_calendar() -> None:
    design = load_discovery_design(_tracked_design(11), require_current=True)
    candidates = discovery_candidates(design)

    assert design["design_sha256"] == (
        "c7cfe43512104388577fc3730a6963f19253b800088eec70c3e18573d1ac5d64"
    )
    assert design["change_control"]["implementation_commit"] == (
        "745cdb6062e0a8b6a26950053dd9db844e1b0806"
    )
    assert len(design["change_control"]["implementation_files_sha256"]) == 7
    assert design["data"]["expected_split_days"]["selection"] == {
        "start_date": "2023-09-14",
        "end_date": "2023-09-17",
        "day_count": 4,
    }
    assert len(candidates) == 12
    assert candidates[0]["candidate_id"] == "conservative-h300"
    assert candidates[-1]["candidate_id"] == "aggressive-h1800"


def test_round_eleven_design_rejects_consumed_registry_tampering(tmp_path) -> None:
    payload = json.loads(_tracked_design(11).read_text(encoding="utf-8"))
    payload["data"]["consumed_registry"] = "registry.json"
    canonical = dict(payload)
    canonical.pop("design_sha256")
    payload["design_sha256"] = _canonical_sha256(canonical)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(payload), encoding="utf-8")

    registry_path = _tracked_design(11).with_name(
        "consumed-periods-through-round-010.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["records"][0]["windows"][0]["start_date"] = "2024-03-14"
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="registry binding"):
        load_discovery_design(design_path)


def test_action_value_charts_fit_twelve_candidates_and_high_auc_values() -> None:
    rows = []
    for horizon in (300, 600, 900, 1800):
        for risk in ("conservative", "regular", "aggressive"):
            rows.append(
                {
                    "candidate_id": f"{risk}-h{horizon}",
                    "fit_status": "trained",
                    "selection_long_auc": 0.798,
                    "selection_short_auc": 0.764,
                    "mean_long_net_bps": -12.2,
                    "mean_short_net_bps": -12.0,
                }
            )

    forecast = _forecast_svg(rows, round_number=11)
    economics = _economics_svg(rows, round_number=11)

    assert _compact_candidate_label("conservative-h1800") == "C 1800s"
    assert ">0.9</text>" in forecast
    assert "conservative-h300" not in forecast
    assert "conservative-h300" not in economics
    assert forecast.count(">C 300s</text>") == 1
    assert economics.count(">A 1800s</text>") == 1
