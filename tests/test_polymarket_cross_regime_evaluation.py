"""Adversarial tests for Polymarket cross-regime promotion evidence."""

from __future__ import annotations

from copy import deepcopy

import pytest

from polymarket_live_support import build_cross_regime_evaluation_fixture
from simple_ai_trading.polymarket_cross_regime_evaluation import (
    PolymarketCrossRegimeEvaluation,
    POLYMARKET_REQUIRED_REGIME_SLICES,
    polymarket_cross_regime_evaluation_sha256,
    validate_polymarket_cross_regime_evaluation,
)


MODEL_SHA = "2" * 64
SOURCE_COMMIT = "b" * 40


def _valid_payload() -> dict[str, object]:
    return build_cross_regime_evaluation_fixture(
        model_artifact_sha256=MODEL_SHA,
        source_commit=SOURCE_COMMIT,
        created_at_ms=1_800_000_000_000,
        market_variant="fiveminute",
    )


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["evaluation_sha256"] = polymarket_cross_regime_evaluation_sha256(payload)
    return payload


def _validate(payload: dict[str, object]) -> PolymarketCrossRegimeEvaluation:
    return validate_polymarket_cross_regime_evaluation(
        payload,
        expected_model_artifact_sha256=MODEL_SHA,
        expected_source_commit=SOURCE_COMMIT,
        expected_market_variant="fiveminute",
        expected_risk_profile="conservative",
    )


def test_complete_cross_regime_fixture_is_model_and_risk_bound() -> None:
    """Accept only the complete, internally consistent synthetic fixture."""
    evaluation = _validate(_valid_payload())

    assert tuple(item.slice_id for item in evaluation.slices) == (
        POLYMARKET_REQUIRED_REGIME_SLICES
    )
    assert evaluation.aggregate.passed is True
    assert evaluation.aggregate.unknown_execution_state_count == 0
    actions = {item.slice_id: item.policy_action for item in evaluation.slices}
    assert actions["liquidity:stale_or_missing_book"] == "abstain"
    assert actions["execution:no_fill_or_unknown_state"] == "abstain"


@pytest.mark.parametrize(
    "missing_slice",
    (
        "direction:bullish",
        "direction:bearish",
        "path:choppy",
        "volatility:stress_volatility",
        "execution:severe_latency",
    ),
)
def test_missing_required_slice_fails_closed(missing_slice: str) -> None:
    """Reject promotion evidence that omits a required market condition."""
    payload = deepcopy(_valid_payload())
    slices = payload["slices"]
    assert isinstance(slices, list)
    payload["slices"] = [item for item in slices if item["slice_id"] != missing_slice]

    with pytest.raises(ValueError, match="required cross-regime slices"):
        _validate(_rehash(payload))


def test_loss_making_trade_slice_cannot_be_declared_passed() -> None:
    """Reject a trade slice whose after-cost result is negative."""
    payload = deepcopy(_valid_payload())
    slices = payload["slices"]
    assert isinstance(slices, list)
    slices[0]["after_cost_net_pnl_quote"] = "-0.01"

    with pytest.raises(ValueError, match="slice failed: direction:bullish"):
        _validate(_rehash(payload))


def test_abstention_slice_cannot_open_a_position() -> None:
    """Reject an abstention claim that exposed capital anyway."""
    payload = deepcopy(_valid_payload())
    slices = payload["slices"]
    assert isinstance(slices, list)
    stale = next(
        item for item in slices if item["slice_id"] == "liquidity:stale_or_missing_book"
    )
    stale["opened_position_count"] = 1

    with pytest.raises(
        ValueError,
        match="slice failed: liquidity:stale_or_missing_book",
    ):
        _validate(_rehash(payload))


def test_unknown_execution_or_untracked_inventory_fails_closed() -> None:
    """Reject any slice with unknown order state or untracked inventory."""
    payload = deepcopy(_valid_payload())
    slices = payload["slices"]
    assert isinstance(slices, list)
    slices[0]["unknown_execution_state_count"] = 1
    slices[0]["untracked_inventory_count"] = 1

    with pytest.raises(ValueError, match="slice failed: direction:bullish"):
        _validate(_rehash(payload))


def test_role_overlap_and_unsealed_test_are_rejected() -> None:
    """Reject selection evidence that can leak or reuse the sealed test role."""
    payload = deepcopy(_valid_payload())
    roles = payload["role_evidence"]
    assert isinstance(roles, dict)
    roles["tune_sha256"] = roles["train_sha256"]
    roles["test_sealed_before_selection"] = False

    with pytest.raises(ValueError, match="train/tune/test isolation failed"):
        _validate(_rehash(payload))


def test_profit_concentration_cannot_be_spoofed() -> None:
    """Recompute concentration from slice P&L instead of trusting the report."""
    payload = deepcopy(_valid_payload())
    aggregate = payload["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["maximum_profit_concentration_fraction"] = "0.01"

    with pytest.raises(ValueError, match="cross-regime aggregate failed"):
        _validate(_rehash(payload))


def test_evaluation_must_bind_the_exact_promoted_model() -> None:
    """Reject a passing report copied from another model artifact."""
    with pytest.raises(ValueError, match="model hash differs"):
        validate_polymarket_cross_regime_evaluation(
            _valid_payload(),
            expected_model_artifact_sha256="3" * 64,
            expected_source_commit=SOURCE_COMMIT,
            expected_market_variant="fiveminute",
            expected_risk_profile="conservative",
        )


def test_evaluation_never_grants_trading_authority() -> None:
    """Keep evidence and order authority as separate capabilities."""
    payload = deepcopy(_valid_payload())
    authority = payload["authority"]
    assert isinstance(authority, dict)
    authority["live_trading_authority"] = True

    with pytest.raises(ValueError, match="cannot grant trading authority"):
        _validate(_rehash(payload))
