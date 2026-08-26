from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "complete-set-holding-yield-reconciliation-v3-2026-08-26.json"
)
TOOL_PATH = ROOT / "tools" / "capture_polymarket_complete_set_holding_yield.py"
EXPECTED_RESULT_HASH = "48e31f3d6021d28946fa1f143f65ff0f6baf9a222424f41e76c2d89875796abe"
EXPECTED_TOOL_HASH = "78e00af5ac6369b49a00c90f200912d2958724602b2925a93d5594e3720bfd1b"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_bytes())


def test_holding_yield_artifact_and_implementation_are_source_bound() -> None:
    artifact = _load()
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"]["sha256"] == EXPECTED_TOOL_HASH
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH
    assert artifact["authority"] == {
        "credentials_used": False,
        "funds_used": False,
        "live_trading_authority": False,
        "orders_placed": False,
        "transactions_sent": False,
    }


def test_complete_set_is_equal_mergeable_and_the_only_current_eligible_value() -> None:
    artifact = _load()
    complete_set = artifact["current_complete_set"]
    assert complete_set["current_holding_reward_condition_ids"] == [
        "0x024b68f77bfc019341ee3db8f57c103334e4b9430bba4746d8c94aafd8b36fee"
    ]
    assert complete_set["current_holding_reward_position_value_pusd"] == "150.00"
    assert complete_set["yes_shares"] == complete_set["no_shares"] == "150"
    assert complete_set["mergeable"] is True
    positions = artifact["position_rows"]
    assert {row["outcome"] for row in positions} == {"Yes", "No"}
    assert sum(Decimal(row["current_value_pusd"]) for row in positions) == Decimal(
        "150"
    )
    assert all(row["mergeable"] for row in positions)


def test_every_observed_daily_yield_reconciles_and_matches_current_rate() -> None:
    artifact = _load()
    observation = artifact["observation"]
    payouts = observation["payouts"]
    assert observation["daily_payout_count"] == 14
    assert observation["positive_daily_payout_count"] == 14
    assert observation["receipt_reconciliation_count"] == 14
    assert observation["sources"]["all_receipts_reconciled"] is True
    assert observation["sources"]["total_account_activity_row_count"] == 14
    assert observation["sources"]["non_yield_account_activity_row_count"] == 0
    assert sum(Decimal(row["amount_pusd"]) for row in payouts) == Decimal("0.1816")
    assert len({row["transaction_hash"] for row in payouts}) == 14
    assert {row["transfer_from"] for row in payouts} == {
        "0x607c8c9866ef3b4665c5a384188706be738d8bf8"
    }
    assert {row["token"] for row in payouts} == {
        "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
    }
    assert observation["implied_sampled_hours"] == 328
    assert observation["sample_count_histogram"] == {
        "21": 1,
        "22": 1,
        "23": 3,
        "24": 9,
    }
    rate = observation["rate_adjudication"]
    assert rate["official_current_rate"] == "0.0325"
    assert rate["all_14_payouts_match_one_integer_hour_count_at_current_rate"] is True
    assert rate["all_14_payouts_match_one_integer_hour_count_at_older_rate"] is False
    assert rate["conflict_resolved_by_current_official_terms_and_realized_payouts"] is True


def test_verdict_is_scoped_to_idle_on_platform_capital_and_not_deployment() -> None:
    artifact = _load()
    assert artifact["source_correction"]["holding_activity_type"] == "YIELD"
    assert artifact["source_correction"]["generic_reward_activity_type"] == "REWARD"
    assert artifact["economics"] == {
        "after_alternative_yield_edge_proven": False,
        "complete_set_redemption_or_merge_value_pusd": "150",
        "deployment_ready": False,
        "direct_relayer_split_merge_user_gas_pusd": "0",
        "market_direction_exposure_of_equal_complete_set": (
            "zero_at_resolution_before_operational_and_custody_risk"
        ),
        "observed_reward_positive": True,
        "validated_gross_edge_for_existing_idle_on_platform_pusd": True,
    }
    verdict = artifact["verdict"]
    assert verdict["accepted_structural_edge"] is True
    assert verdict["future_profit_guaranteed"] is False
    assert verdict["deployment_authorized"] is False
