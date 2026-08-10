from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_AI_CANDIDATE_IDS,
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_IDS,
    load_round25_candidate_amendment,
    load_round25_candidate_design,
    validate_round25_candidate_design,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-twap-native-candidate-selection-design-v1.json"
)
AMENDMENT_PATH = DESIGN_PATH.with_name(
    "round-025-twap-native-candidate-selection-amendment-v2.json"
)


def _design() -> dict[str, object]:
    return json.loads(DESIGN_PATH.read_text(encoding="utf-8"))


def test_candidate_design_loads_with_exact_finite_ledgers() -> None:
    design = load_round25_candidate_design(DESIGN_PATH)

    assert design["design_sha256"] == POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    assert tuple(
        item["candidate_id"] for item in design["finite_candidate_ledger"]
    ) == POLYMARKET_ROUND25_CANDIDATE_IDS
    assert tuple(
        item["candidate_id"] for item in design["ai_candidate_audit"]["candidates"]
    ) == POLYMARKET_ROUND25_AI_CANDIDATE_IDS
    assert design["ai_candidate_audit"]["rejected_candidates"][0][
        "candidate_id"
    ] == "fino1-8b"
    assert not any(design["truth_state"].values())


def test_amendment_separates_development_from_future_sealed_test() -> None:
    amendment = load_round25_candidate_amendment(AMENDMENT_PATH)

    assert amendment["design_sha256"] == POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    assert amendment["parent_candidate_design_sha256"] == (
        POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    )
    assert amendment["development_campaign"]["roles"] == [
        "train",
        "calibration",
        "selection",
    ]
    assert amendment["development_campaign"]["test_role_present"] is False
    assert amendment["separate_sealed_test_campaign"]["required"] is True
    assert amendment["separate_sealed_test_campaign"]["plan_available"] is False
    assert amendment["separate_sealed_test_campaign"]["minimum_conditions"] == 500
    assert not any(amendment["truth_state"].values())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("feature_contracts", "joint_width"), 149),
        (("feature_contracts", "structural_probability_allowed"), True),
        (("population_and_splits", "target"), "constructed_twap_target"),
        (("population_and_splits", "training_endpoints_per_condition"), 1200),
        (("candidate_selection", "market_prior_control_must_be_beaten"), False),
        (("ai_candidate_audit", "direct_250ms_trade_generation_allowed"), True),
        (("ai_candidate_audit", "entry_expansion_allowed"), True),
        (("execution_gate", "midpoint_or_zero_latency_execution_allowed"), True),
        (("truth_state", "profitability_claim"), True),
    ],
)
def test_rehashed_policy_changes_still_fail_expected_design_identity(
    path: tuple[str, str],
    replacement: object,
) -> None:
    design = deepcopy(_design())
    design[path[0]][path[1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="design differs"):
        validate_round25_candidate_design(design)


def test_candidate_addition_is_rejected_even_without_rehashing() -> None:
    design = deepcopy(_design())
    design["finite_candidate_ledger"].append(  # type: ignore[union-attr]
        {"candidate_id": "unregistered-transformer", "family": "sequence_ml"}
    )

    with pytest.raises(ValueError, match="design differs"):
        validate_round25_candidate_design(design)


def test_loader_rejects_duplicate_keys_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"round":25,"round":26}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        load_round25_candidate_design(duplicate)

    link = tmp_path / "design-link.json"
    try:
        link.symlink_to(DESIGN_PATH)
    except OSError:
        pytest.skip("host does not permit test symlinks")
    with pytest.raises(ValueError, match="unavailable"):
        load_round25_candidate_design(link)
