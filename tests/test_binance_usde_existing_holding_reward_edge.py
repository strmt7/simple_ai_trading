from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-usde-existing-holding-reward-edge-v1-2026-08-26.json"
)
REGISTRY_PATH = (
    ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_HASH = (
    "4640635514ad43ed846660c204a95c0d59ed75ac3ccbf5f17a0b70f3d5726f6a"
)
EXPECTED_REGISTRY_HASH = (
    "a661ba68ecdf87eaece27616e5ede3ca0864ca844b3f9553d7d0fd4a59f645f8"
)


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


def test_usde_edge_is_hash_bound_scoped_and_non_trading() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_RESULT_HASH
    assert _canonical_hash(artifact) == EXPECTED_RESULT_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": True,
        "accepted_scope": (
            "current_same_asset_gross_USDe_reward_only_for_eligible_USDe_"
            "already_independently_held_on_Binance_for_at_least_24_hours_"
            "without_acquisition_or_balance_retention_for_the_reward"
        ),
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "status": (
            "accepted_scoped_current_automatic_USDe_holding_reward_account_"
            "eligibility_and_owned_distribution_unproved"
        ),
        "trading_authority": False,
    }
    assert artifact["authority"] == {
        "account_state_accessed": False,
        "credentials_used": False,
        "funded_actions": 0,
        "orders_or_transfers_submitted": 0,
        "public_read_only_research": True,
    }


def test_usde_current_same_token_reward_math_reconstructs() -> None:
    artifact = _load(ARTIFACT_PATH)
    economics = artifact["economics"]
    apr = Decimal(economics["current_reference_apr_percent"]) / Decimal(100)
    daily_fraction = apr / Decimal(365)

    assert daily_fraction == Decimal(economics["daily_reward_fraction"])
    assert daily_fraction * Decimal(10_000) == Decimal(
        economics["daily_reward_bips_of_eligible_balance"]
    )
    assert daily_fraction * Decimal(1_000) == Decimal(
        economics["daily_reward_usde_per_1000_eligible_usde"]
    )
    assert apr * Decimal(7) / Decimal(365) * Decimal(10_000) == Decimal(
        economics["seven_day_reward_bips_of_eligible_balance"]
    )
    assert apr * Decimal(7) / Decimal(365) * Decimal(1_000) == Decimal(
        economics["seven_day_reward_usde_per_1000_eligible_usde"]
    )
    assert "no stablecoin conversion or fiat parity" in economics["same_asset_unit"]


def test_usde_terms_require_existing_eligible_balance_and_preserve_risk() -> None:
    artifact = _load(ARTIFACT_PATH)
    terms = artifact["eligibility_and_terms"]

    assert terms["campaign_period"] == {
        "end": "until_further_notice",
        "start_utc": "2025-09-22T00:00:00Z",
    }
    assert terms["minimum_eligible_balance_usde"] == "0.01"
    assert terms["minimum_holding_duration_hours_before_accrual"] == 24
    assert terms["no_daily_reward_limit"] is True
    assert set(terms["eligible_accounts"]) == {
        "Spot",
        "Funding",
        "Futures",
        "Margin",
        "master account",
        "sub-accounts",
    }
    assert terms["identity_verification_required"] is True
    assert terms["jurisdiction_and_user_status_restrictions_apply"] is True
    assert terms["reward_claim_required"] is False
    assert "random daily time" in terms["snapshot"]
    assert "no exact payout-time claim" in terms["reward_distribution_time_conflict"]

    prohibited = " ".join(artifact["prohibited_actions"])
    assert "acquiring depositing converting borrowing or retaining USDe" in prohibited
    assert "guaranteed persistent fiat return" in prohibited
    assert "double-counting collateral efficiency" in prohibited
    limitations = " ".join(artifact["limitations"])
    assert "deviate from its reference value" in limitations
    assert "can be changed weekly" in limitations


def test_usde_primary_sources_and_registry_lineage_are_exact() -> None:
    artifact = _load(ARTIFACT_PATH)
    sources = artifact["sources"]
    assert sources[0]["url"] == "https://www.binance.com/en/earn/usde-page"
    assert sources[0]["normalized_live_visible_page_characters"] == 6287
    assert sources[0]["normalized_live_visible_page_sha256"] == (
        "4ca89a40bbb750bb07ae9d0876e395fad2b06ed26cc026067af193ee2f962a02"
    )
    assert sources[1]["effective_date"] == "2026-01-05"
    assert sources[1]["version"] == "1.0"
    assert sources[1]["url"].endswith(
        "fced23db7e53b5667fcc2fbdd06acc5b5d949f3a2f3bcea17b85c337a85b0e44.pdf"
    )
    assert sources[2]["article_code"] == "ca03b33b8dba44669a3f685d2d1c0ccb"
    assert sources[2]["normalized_live_visible_page_sha256"] == (
        "8f1d46f46e4ec00a1bf1bce11d0b7c4cdba37ce4a4074e124923523bc0975bab"
    )

    registry = _load(REGISTRY_PATH)
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _canonical_hash(registry) == EXPECTED_REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "same_account_stable_value_yield_allocation"
    )
    assert family["priority_rank"] == 8
    assert {
        "path": (
            "docs/model-research/action-value/"
            "binance-usde-existing-holding-reward-edge-v1-2026-08-26.json"
        ),
        "result_sha256": EXPECTED_RESULT_HASH,
    } in family["canonical_artifacts"]
    assert "usde" in family["venue_scope"]
