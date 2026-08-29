from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
BUILDER_PATH = ACTION_VALUE / (
    "polymarket-organic-third-party-builder-fee-overlay-v1-2026-08-26.json"
)
BFUSD_PATH = ACTION_VALUE / (
    "binance-bfusd-existing-holding-reward-unit-conflict-gate-v1-2026-08-26.json"
)
SMART_ARBITRAGE_PATH = ACTION_VALUE / (
    "binance-smart-arbitrage-terminal-family-adjudication-v1-2026-08-26.json"
)
BUILDER_HASH = "8c070b6a4b07070ffdd5ba703da1ca3788faffcb4d748633a18269dc02c17885"
BFUSD_HASH = "54fe3d3e23a92290debdc67d1e7e19ecac6c06441c045f1aa21fe3e62558c03c"
SMART_ARBITRAGE_HASH = (
    "03b652fcd7e50c0671abbfb73f68f69509a2e5d7f75d8166f6b74743eab630d3"
)
REGISTRY_HASH = "e8c32ad724da73148aa1becc77fe413a243e11fa8f444d514b10e844f9089bfe"


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


def test_builder_fee_overlay_is_hash_bound_scoped_and_non_trading() -> None:
    artifact = _load(BUILDER_PATH)

    assert artifact["result_sha256"] == BUILDER_HASH
    assert _canonical_hash(artifact) == BUILDER_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": True,
        "accepted_scope": (
            "current_gross_pUSD_builder_fee_only_on_bona_fide_independently_"
            "existing_third_party_orders_routed_through_an_owned_application_"
            "with_an_account_confirmed_active_builder_code_and_explicitly_"
            "configured_disclosed_positive_fee_rate"
        ),
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "status": (
            "accepted_scoped_direction_independent_incremental_builder_fee_"
            "overlay_account_eligibility_organic_external_flow_and_owned_"
            "payout_unproved"
        ),
        "trading_authority": False,
    }
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_transfers_submitted"] == 0


def test_builder_fee_formula_and_organic_flow_guards_reconstruct() -> None:
    artifact = _load(BUILDER_PATH)
    economics = artifact["economics"]
    examples = economics["gross_pusd_per_1000_matched_pusd"]

    for rate_key, rate in (
        ("at_1_bp", Decimal(1)),
        ("at_10_bps", Decimal(10)),
        ("at_50_bps_maker_maximum", Decimal(50)),
        ("at_100_bps_taker_maximum", Decimal(100)),
    ):
        assert Decimal(1000) * rate / Decimal(10000) == Decimal(examples[rate_key])

    assert economics["maker_rate_limit_bps"] == 50
    assert economics["taker_rate_limit_bps"] == 100
    assert economics["rate_granularity_bps"] == 1
    assert "no stablecoin parity" in economics["unit_binding"]
    assert "current tier comparison" in artifact["eligibility_and_terms"][
        "tier_eligibility_conflict"
    ].lower()
    prohibited = " ".join(artifact["prohibited_actions"])
    assert "self-referred" in prohibited
    assert "Silently charging".lower() in prohibited.lower()
    assert "gross collected fees after-cost profit" in prohibited


def test_bfusd_reward_unit_conflict_fails_closed() -> None:
    artifact = _load(BFUSD_PATH)

    assert artifact["result_sha256"] == BFUSD_HASH
    assert _canonical_hash(artifact) == BFUSD_HASH
    assert artifact["adjudication"]["accepted_edge"] is False
    conflict = artifact["reward_asset_conflict"]
    assert "in BFUSD" in conflict["current_product_page"]
    assert "USD stablecoin" in conflict["current_rate_faq"]
    assert "USD Stablecoin" in conflict["governing_terms_effective_2026_01_05"]
    assert conflict["same_unit_forward_reward_floor"] == "0"
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_transfers_submitted"] == 0


def test_smart_arbitrage_is_terminal_same_family_without_subsidy() -> None:
    artifact = _load(SMART_ARBITRAGE_PATH)

    assert artifact["result_sha256"] == SMART_ARBITRAGE_HASH
    assert _canonical_hash(artifact) == SMART_ARBITRAGE_HASH
    assert artifact["adjudication"]["accepted_edge"] is False
    assert "spot_perpetual_funding_carry" in artifact["adjudication"]["status"]
    hurdle = artifact["break_even_contract"]["required_inequality"]
    assert "cumulative_received_funding >" in hurdle
    assert "spot_maker_entry_fee" in hurdle
    assert "futures_taker_exit_fee" in hurdle
    assert artifact["break_even_contract"]["public_current_after_cost_floor"] == "0"
    assert artifact["authority"]["funded_actions"] == 0


def test_registry_binds_new_edge_conflict_and_terminal_adjudication() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 20
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 43))

    builder = next(
        row
        for row in hypotheses
        if row["mechanism"] == "polymarket_organic_third_party_builder_fee_overlay"
    )
    assert builder["priority_rank"] == 24
    assert builder["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-organic-third-party-builder-fee-overlay-v1-2026-08-26.json"
            ),
            "result_sha256": BUILDER_HASH,
        }
    ]

    stable_yield = next(
        row
        for row in hypotheses
        if row["mechanism"] == "same_account_stable_value_yield_allocation"
    )
    assert {
        "path": (
            "docs/model-research/action-value/"
            "binance-bfusd-existing-holding-reward-unit-conflict-gate-v1-2026-08-26.json"
        ),
        "result_sha256": BFUSD_HASH,
    } in stable_yield["canonical_artifacts"]

    terminal = {
        row["family"]: row["canonical_result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    }
    assert terminal[
        "binance_smart_arbitrage_packaged_spot_perpetual_funding_carry"
    ] == SMART_ARBITRAGE_HASH
