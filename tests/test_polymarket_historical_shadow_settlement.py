from __future__ import annotations

import hashlib
import json

import pytest

from simple_ai_trading.polymarket_historical_shadow_settlement import (
    settle_shadow_opportunity,
    validate_shadow_official_resolution,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _opportunity() -> dict[str, object]:
    quote = {
        "outcome": "Up",
        "token_id": "1",
        "quantity": "5",
        "filled_quantity": "5",
        "average_price": "0.6",
        "worst_price": "0.6",
        "notional_quote": "3",
        "fee_quote": "0.08",
        "total_cost_quote": "3.08",
        "total_cost_per_share": "0.616",
        "displayed_fillable": True,
        "book_source_time_ms": 1_000_089,
        "book_payload_sha256": "a" * 64,
    }
    down = {
        **quote,
        "outcome": "Down",
        "token_id": "2",
        "average_price": "0.4",
        "worst_price": "0.4",
        "notional_quote": "2",
        "total_cost_quote": "2.08",
        "total_cost_per_share": "0.416",
        "book_payload_sha256": "b" * 64,
    }
    body: dict[str, object] = {
        "schema_version": "polymarket-btc-shadow-opportunity-v1",
        "status": "candidate",
        "reason": "",
        "gamma_market_id": "42",
        "condition_id": "0x" + "3" * 64,
        "slug": "btc-updown-5m-1000",
        "gamma_payload_sha256": "c" * 64,
        "event_start_ms": 1_000_000,
        "event_end_ms": 1_300_000,
        "decision_time_ms": 1_090_000,
        "observed_at_ms": 1_090_100,
        "quote_observation_latency_ms": 100,
        "probability_up": "0.75",
        "selected_outcome": "Up",
        "expected_terminal_value_per_share": "0.134",
        "minimum_required_edge_per_share": "0.01",
        "maximum_loss_quote": "3.08",
        "up_quote": quote,
        "down_quote": down,
        "model_candidate_id": "candidate",
        "model_pretest_sha256": "d" * 64,
        "model_evaluation_sha256": "e" * 64,
        "model_support_profile_sha256": "f" * 64,
        "outside_training_range_count": 0,
        "extreme_outlier_count": 0,
        "clob_market_info_sha256": "1" * 64,
        "tick_size": "0.01",
        "minimum_order_size": "5",
        "fee_rate": "0.07",
        "fee_exponent": 1,
        "general_order_delay_seconds": 0,
        "taker_order_delay_enabled": True,
        "trading_authority": False,
        "proposal_authority": False,
        "execution_or_profitability_claim": False,
    }
    return {**body, "artifact_sha256": _canonical_sha(body)}


def _gamma(*, up_wins: bool = True) -> dict[str, object]:
    return {
        "id": "42",
        "conditionId": "0x" + "3" * 64,
        "slug": "btc-updown-5m-1000",
        "closed": True,
        "acceptingOrders": False,
        "endDate": "1970-01-01T00:21:40Z",
        "outcomes": "[\"Up\",\"Down\"]",
        "outcomePrices": "[\"1\",\"0\"]" if up_wins else "[\"0\",\"1\"]",
    }


def _clob(*, up_wins: bool = True) -> dict[str, object]:
    return {
        "condition_id": "0x" + "3" * 64,
        "closed": True,
        "accepting_orders": False,
        "tokens": [
            {
                "outcome": "Up",
                "token_id": "1",
                "price": 1 if up_wins else 0,
                "winner": up_wins,
            },
            {
                "outcome": "Down",
                "token_id": "2",
                "price": 0 if up_wins else 1,
                "winner": not up_wins,
            },
        ],
    }


def test_joint_terminal_sources_create_counterfactual_only() -> None:
    artifact, artifact_sha = settle_shadow_opportunity(
        _opportunity(),
        gamma_market=_gamma(),
        clob_market=_clob(),
        resolution_observed_at_ms=1_301_000,
        source_log_sha256="9" * 64,
        source_commit="8" * 40,
    )
    assert artifact["artifact_sha256"] == artifact_sha
    assert artifact["official_resolution"]["winner"] == "Up"
    assert artifact["prediction"]["selected_outcome_won"] is True
    counterfactual = artifact["displayed_depth_counterfactual"]
    assert counterfactual["net_pnl_quote"] == "1.92"
    assert counterfactual["real_order_submitted"] is False
    assert counterfactual["real_fill_observed"] is False
    assert artifact["trading_authority"] is False
    assert artifact["profitability_claim"] is False


def test_disagreeing_terminal_sources_are_rejected() -> None:
    with pytest.raises(ValueError, match="outcomes disagree"):
        settle_shadow_opportunity(
            _opportunity(),
            gamma_market=_gamma(up_wins=True),
            clob_market=_clob(up_wins=False),
            resolution_observed_at_ms=1_301_000,
            source_log_sha256="9" * 64,
            source_commit="8" * 40,
        )


def test_unsettled_gamma_is_rejected() -> None:
    gamma = _gamma()
    gamma["closed"] = False
    with pytest.raises(ValueError, match="Gamma terminal"):
        settle_shadow_opportunity(
            _opportunity(),
            gamma_market=gamma,
            clob_market=_clob(),
            resolution_observed_at_ms=1_301_000,
            source_log_sha256="9" * 64,
            source_commit="8" * 40,
        )


def test_abstained_opportunity_can_supply_resolution_identity() -> None:
    opportunity = _opportunity()
    opportunity["status"] = "abstain"
    opportunity["reason"] = "after_cost_edge_below_threshold"
    opportunity.pop("artifact_sha256")
    opportunity["artifact_sha256"] = _canonical_sha(opportunity)
    resolution = validate_shadow_official_resolution(
        opportunity,
        gamma_market=_gamma(),
        clob_market=_clob(),
        resolution_observed_at_ms=1_301_000,
    )
    assert resolution["winner"] == "Up"
    assert resolution["sources_agree"] is True
    assert resolution["trading_authority"] is False
