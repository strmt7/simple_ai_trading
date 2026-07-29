from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_historical_shadow_campaign import (
    build_shadow_hour_evaluation,
    load_shadow_hour_policy,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _policy(path: Path) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "polymarket-btc-shadow-hour-policy-v1",
        "created_at_ms": 900_000,
        "eligible_event_start_ms": 1_000_000,
        "expected_run_end_ms": 2_000_000,
        "scope": {
            "venue": "polymarket",
            "asset": "BTC",
            "market_variant": "fiveminute",
            "source_log_basename": path.name,
            "public_data_only": True,
            "credentials_used": False,
            "orders_submitted": False,
        },
        "model_identity": {
            "candidate_id": "candidate",
            "dataset_sha256": "1" * 64,
            "pretest_artifact_sha256": "2" * 64,
            "evaluation_artifact_sha256": "3" * 64,
            "support_profile_sha256": "4" * 64,
        },
        "selection_policy": {
            "maximum_selected_entries_per_event": 1,
            "selection": "first_chronological_status_candidate",
            "threshold_changes_after_freeze_allowed": False,
            "outcomes_or_future_books_consulted": False,
        },
        "settlement_policy": {
            "gamma_and_clob_terminal_identity_required": True,
            "gamma_and_clob_winner_agreement_required": True,
            "displayed_depth_fill_is_counterfactual_only": True,
            "real_fill_claim_allowed": False,
        },
        "reporting_policy": {
            "minimum_events_for_profitability_claim": 50,
            "profitability_claim_for_this_run_allowed": False,
        },
        "trading_authority": False,
        "profitability_claim": False,
    }
    return {**body, "policy_sha256": _sha(body)}


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
        "book_source_time_ms": 1_089_000,
        "book_payload_sha256": "5" * 64,
    }
    down = {
        **quote,
        "outcome": "Down",
        "token_id": "2",
        "total_cost_quote": "2.08",
        "total_cost_per_share": "0.416",
        "book_payload_sha256": "6" * 64,
    }
    body = {
        "schema_version": "polymarket-btc-shadow-opportunity-v1",
        "status": "candidate",
        "reason": "",
        "gamma_market_id": "42",
        "condition_id": "0x" + "7" * 64,
        "slug": "btc-updown-5m-1000",
        "gamma_payload_sha256": "8" * 64,
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
        "model_pretest_sha256": "2" * 64,
        "model_evaluation_sha256": "3" * 64,
        "model_support_profile_sha256": "4" * 64,
        "outside_training_range_count": 0,
        "extreme_outlier_count": 0,
        "clob_market_info_sha256": "9" * 64,
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
    return {**body, "artifact_sha256": _sha(body)}


def _terminal() -> tuple[dict[str, object], dict[str, object], int]:
    gamma = {
        "id": "42",
        "conditionId": "0x" + "7" * 64,
        "slug": "btc-updown-5m-1000",
        "closed": True,
        "acceptingOrders": False,
        "endDate": "1970-01-01T00:21:40Z",
        "outcomes": "[\"Up\",\"Down\"]",
        "outcomePrices": "[\"1\",\"0\"]",
    }
    clob = {
        "condition_id": "0x" + "7" * 64,
        "closed": True,
        "accepting_orders": False,
        "tokens": [
            {"outcome": "Up", "token_id": "1", "price": 1, "winner": True},
            {"outcome": "Down", "token_id": "2", "price": 0, "winner": False},
        ],
    }
    return gamma, clob, 1_301_000


def test_frozen_hour_selects_first_candidate_without_authority(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "shadow.jsonl"
    policy = _policy(policy_path)
    opportunity = _opportunity()
    prediction = {
        "status": "observed",
        "event_start_ms": 1_000_000,
        "decision_time_ms": 1_090_000,
        "probability_up": 0.75,
    }
    records = (
        {"event": "prediction", "payload": prediction},
        {"event": "opportunity", "payload": opportunity},
        {
            "event": "stopped",
            "payload": {
                "running": False,
                "last_errors": {"spot": "", "perpetual": ""},
            },
        },
    )
    artifact, _ = build_shadow_hour_evaluation(
        policy=policy,
        records=records,
        source_log_sha256="a" * 64,
        terminal_sources={1_000_000: _terminal()},
        source_commit="b" * 40,
    )
    counterfactual = artifact["first_candidate_counterfactual"]
    assert counterfactual["selected_events"] == 1
    assert counterfactual["winning_events"] == 1
    assert counterfactual["net_pnl_quote"] == "1.92"
    assert counterfactual["real_fills_observed"] == 0
    assert artifact["profitability_claim"] is False
    assert artifact["trading_authority"] is False


def test_policy_rejects_rehashed_threshold_drift(tmp_path: Path) -> None:
    policy_path = tmp_path / "shadow.jsonl"
    policy = _policy(policy_path)
    policy["selection_policy"]["maximum_selected_entries_per_event"] = 2
    policy.pop("policy_sha256")
    policy["policy_sha256"] = _sha(policy)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="semantics differ"):
        load_shadow_hour_policy(policy_path)
