from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
REFERRAL_PATH = ACTION_VALUE / (
    "polymarket-organic-referral-net-fee-overlay-v1-2026-08-26.json"
)
FLEXIBLE_LOAN_PATH = ACTION_VALUE / (
    "binance-flexible-loan-simple-earn-collateral-yield-gate-v1-2026-08-26.json"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REFERRAL_HASH = "f7aec4a5340cba42abb120a43cda1ed1fa4d5b03632b3c062c0d00d7b5636cf0"
FLEXIBLE_LOAN_HASH = (
    "ac010265c5236152907ac7b3c12ce13104f473b4cc61c5db43fb8b28c6678182"
)
REGISTRY_HASH = "afa26a57c9ca4525021ef1d728993ecc52a427ac03e8ee3f48bd15ab0203bf71"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_referral_overlay_is_hash_bound_scoped_and_non_trading() -> None:
    artifact = _load(REFERRAL_PATH)

    assert artifact["result_sha256"] == REFERRAL_HASH
    assert _canonical_hash(artifact) == REFERRAL_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": True,
        "accepted_scope": (
            "current_gross_pUSD_referral_reward_only_for_authentic_"
            "independently_acquired_external_users_when_the_referrer_account_"
            "already_independently_exceeds_10000_USD_lifetime_Polymarket_"
            "volume_without_creating_volume_to_qualify"
        ),
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "status": (
            "accepted_scoped_direction_independent_organic_referral_net_fee_"
            "overlay_account_eligibility_referral_attribution_and_owned_"
            "payout_unproved"
        ),
        "trading_authority": False,
    }
    assert artifact["authority"]["referral_links_created"] == 0
    assert artifact["authority"]["orders_or_transfers_submitted"] == 0


def test_referral_rates_caps_and_anti_manufacturing_guards_reconstruct() -> None:
    artifact = _load(REFERRAL_PATH)
    economics = artifact["economics"]
    terms = artifact["eligibility_and_terms"]

    assert Decimal(100) * Decimal(
        economics["direct_referral_rate_of_net_fees"]
    ) == Decimal(economics["direct_reward_pusd_per_100_pusd_net_fees"])
    assert Decimal(100) * Decimal(
        economics["indirect_referral_rate_of_net_fees"]
    ) == Decimal(economics["indirect_reward_pusd_per_100_pusd_net_fees"])
    assert terms["lifetime_trading_volume_threshold_usd"] == "10000"
    assert terms["attribution_window_after_link_click_days"] == 30
    assert terms["reward_end"] == (
        "the earlier of the referred user reaching Platinum or 30 days after signup"
    )
    assert terms["omnibus_third_party_integrations_eligible"] is False
    prohibited = " ".join(artifact["prohibited_actions"])
    assert "self-referral" in prohibited
    assert "manufacturing self-matching wash circular" in prohibited
    assert "double-counting the same fee base" in prohibited


def test_flexible_loan_candidate_fails_closed_without_account_economics() -> None:
    artifact = _load(FLEXIBLE_LOAN_PATH)

    assert artifact["result_sha256"] == FLEXIBLE_LOAN_HASH
    assert _canonical_hash(artifact) == FLEXIBLE_LOAN_HASH
    assert artifact["adjudication"]["accepted_edge"] is False
    assert artifact["break_even_contract"]["current_public_after_cost_floor"] == "0"
    assert artifact["authority"] == {
        "account_state_accessed": False,
        "credentials_present": False,
        "credentials_used": False,
        "funded_actions": 0,
        "signed_requests": 0,
        "orders_or_transfers_submitted": 0,
        "public_read_only_research": True,
    }
    endpoint = artifact["endpoint_contract"]
    assert all(
        item.startswith("GET ")
        for item in endpoint["read_only_signed_user_data_endpoints"]
    )
    assert all(
        item.startswith("POST ")
        for item in endpoint["state_changing_trade_endpoints_not_authorized"]
    )
    assert "realized_collateral_Simple_Earn_rewards" in artifact[
        "break_even_contract"
    ]["required_total_inequality"]


def test_registry_binds_referral_and_flexible_loan_without_overpromotion() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 43))

    referral = next(
        row
        for row in hypotheses
        if row["mechanism"] == "polymarket_organic_referral_net_fee_overlay"
    )
    assert referral["priority_rank"] == 25
    assert referral["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-organic-referral-net-fee-overlay-v1-2026-08-26.json"
            ),
            "result_sha256": REFERRAL_HASH,
        }
    ]

    loan = next(
        row
        for row in hypotheses
        if row["mechanism"]
        == "binance_flexible_loan_simple_earn_collateral_yield_retention"
    )
    assert loan["priority_rank"] == 26
    assert "candidate_not_accepted" in loan["current_status"]
    assert loan["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-flexible-loan-simple-earn-collateral-yield-gate-v1-"
                "2026-08-26.json"
            ),
            "result_sha256": FLEXIBLE_LOAN_HASH,
        }
    ]
