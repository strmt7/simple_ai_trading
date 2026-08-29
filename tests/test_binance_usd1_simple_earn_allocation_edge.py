from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs/model-research/action-value"
    / "binance-usd1-simple-earn-versus-holding-airdrop-allocation-edge-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
ARTIFACT_HASH = "a4158bf059f4f5ad839b2f504c08c4afc65615260b4171533866f4c2337494e0"
REGISTRY_HASH = "da3ddaf82a2cb0929353460a7e09812b47f940e953a3f1da43b04f72a55c8488"


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


def test_usd1_allocation_edge_is_hash_bound_action_free_and_mutually_exclusive() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == ARTIFACT_HASH
    assert _canonical_hash(artifact) == ARTIFACT_HASH
    assert artifact["authority"]["authenticated_requests"] == 0
    assert artifact["authority"]["subscriptions_redemptions_conversions_transfers_orders_or_quotes"] == 0
    terms = artifact["source_bound_terms"]
    assert terms["holding_airdrop_eligible_account_categories_include_Simple_Earn"] is False
    assert terms["same_principal_can_receive_both_rewards"] is False
    assert artifact["accepted_scope"] == {
        "accepted_edge": True,
        "scope": "current fixed 7 percent Simple Earn bonus on at most the first 1500 USD1 already independently held idle when the account is eligible capacity is available the balance is not needed for prompt liquidity and the proved remaining bonus horizon exceeds every transition and opportunity cost",
        "acquisition_or_conversion_edge_accepted": False,
        "same_principal_double_counting_permitted": False,
        "market_direction_forecast_required": False,
        "after_all_external_costs_proved": False,
        "profitability_claim": False,
        "deployment_ready": False,
    }
    assert artifact["execution_gate"]["public_forward_net_profit_floor"] == "0"


def test_usd1_fixed_and_current_rate_allocation_math_reconstructs() -> None:
    artifact = _load(ARTIFACT)
    snapshot = artifact["current_public_snapshot"]
    economics = artifact["economics"]
    principal = Decimal(economics["principal_usd1"])
    full = economics["current_full_rate_comparison"]
    fixed = economics["fixed_bonus_only_comparison"]
    display = Decimal(snapshot["displayed_max_apr_percent"])
    realtime = Decimal(snapshot["displayed_realtime_apr_percent"])
    bonus = Decimal(snapshot["displayed_bonus_apr_percent"])
    airdrop = Decimal(fixed["holding_airdrop_current_apr_percent"])

    assert display == realtime + bonus
    assert snapshot["simultaneous_public_stablecoin_max_apr_percent"] == {
        "USD1": "8.62",
        "U": "8.59",
        "USDC": "7.28",
        "USDT": "6.80",
    }
    assert "cannot justify conversion" in snapshot["cross_token_rule"]
    assert Decimal(full["incremental_apr_percentage_points"]) == display - airdrop
    assert Decimal(fixed["incremental_apr_percentage_points"]) == bonus - airdrop
    assert Decimal(full["incremental_reward_usd1_per_365_days"]) == principal * (
        display - airdrop
    ) / Decimal(100)
    assert Decimal(fixed["incremental_reward_usd1_per_365_days"]) == principal * (
        bonus - airdrop
    ) / Decimal(100)
    assert Decimal(full["maximum_balance_where_capped_bonus_plus_current_realtime_rate_exceeds_current_airdrop_rate_usd1"]) == principal * bonus / (
        airdrop - realtime
    )
    assert Decimal(fixed["maximum_balance_where_capped_fixed_bonus_exceeds_current_airdrop_rate_usd1"]) == principal * bonus / airdrop


def test_immediate_fixed_bonus_sensitivity_reconstructs() -> None:
    artifact = _load(ARTIFACT)
    economics = artifact["economics"]
    sensitivity = economics[
        "immediate_fixed_bonus_versus_wait_through_airdrop_end_sensitivity"
    ]
    principal = Decimal(economics["principal_usd1"])
    bonus = Decimal(artifact["current_public_snapshot"]["displayed_bonus_apr_percent"])
    airdrop = Decimal(
        economics["fixed_bonus_only_comparison"]["holding_airdrop_current_apr_percent"]
    )
    immediate = principal * bonus / Decimal(100) * Decimal(29) / Decimal(365)
    wait_airdrop = principal * airdrop / Decimal(100) * Decimal(8) / Decimal(365)
    delayed = principal * bonus / Decimal(100) * Decimal(21) / Decimal(365)
    advantage = immediate - wait_airdrop - delayed

    assert Decimal(sensitivity["immediate_fixed_bonus_reward_usd1"]) == immediate
    assert Decimal(sensitivity["wait_airdrop_reward_usd1_equivalent"]) == wait_airdrop
    assert Decimal(sensitivity["delayed_fixed_bonus_reward_usd1"]) == delayed
    assert Decimal(sensitivity["immediate_fixed_bonus_advantage_usd1_equivalent"]) == advantage
    assert Decimal(sensitivity["immediate_fixed_bonus_advantage_bps_of_principal"]) == advantage / principal * Decimal(10000)


def test_registry_promotes_only_the_scoped_existing_usd1_allocation() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 21
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "same_account_stable_value_yield_allocation"
    )
    assert {
        "path": "docs/model-research/action-value/binance-usd1-simple-earn-versus-holding-airdrop-allocation-edge-v1-2026-08-27.json",
        "result_sha256": ARTIFACT_HASH,
    } in hypothesis["canonical_artifacts"]
    assert "independently_already_held_idle_USD1" in hypothesis["current_status"]
    assert any(
        "USDT_to_USD1_acquisition" in shortcut
        for shortcut in hypothesis["prohibited_shortcuts"]
    )
