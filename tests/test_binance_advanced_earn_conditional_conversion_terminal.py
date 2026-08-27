from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-advanced-earn-conditional-conversion-terminal-adjudication-v1-"
    "2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "15f160e3d54f0be09611bb36901b1d9061a2a173643c0562996ecb2824320a3f"
REGISTRY_HASH = "d9b8d325ca4099fa6e89b4152ca3ca74284056369bdecc54f61b37e43ed0f652"


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


def test_conditional_conversion_family_is_hash_bound_and_terminal() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "market_direction_forecast_required": True,
        "profitability_claim": False,
        "status": (
            "terminal_direction_dependent_conditional_conversion_family_no_"
            "market_invariant_after_cost_floor_or_distinct_subsidy_proved"
        ),
        "trading_authority": False,
    }
    assert artifact["break_even_contract"]["public_current_after_cost_floor"] == "0"
    assert artifact["family_identity"]["not_exact_exchange_option_equivalence_claim"]


def test_discount_buy_and_dual_investment_downside_cannot_hide_behind_apr() -> None:
    artifact = _load(ARTIFACT)
    mechanism = artifact["official_mechanism"]

    assert mechanism["discount_buy"] == {
        "cannot_cancel_or_redeem_early_after_subscription": True,
        "non_principal_protected_high_yield_structured_product": True,
        "preset_terms": [
            "target buy price",
            "knockout price",
            "APR",
            "settlement date",
        ],
        "settlement_price_at_or_above_knockout": (
            "stablecoin principal plus stated APR reward"
        ),
        "settlement_price_at_or_below_target": (
            "100 percent of stablecoin principal buys crypto at the target buy price"
        ),
        "settlement_price_between_target_and_knockout": (
            "50 percent of stablecoin principal buys crypto at the target buy price "
            "and 50 percent is returned"
        ),
    }
    assert "forgo gains" in mechanism["dual_investment"]["sell_high_upside_cap"]
    assert "worth less" in mechanism["dual_investment"]["buy_low_downside"]
    assert artifact["authority"]["product_or_quote_requests_sent"] == 0
    assert artifact["authority"]["subscriptions_submitted"] == 0


def test_registry_terminalizes_family_without_new_priority_or_acceptance() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 16
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 37)
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_advanced_earn_discount_buy_and_dual_investment_conditional_conversion"
    )
    assert terminal["canonical_result_sha256"] == EXPECTED_HASH
    assert "direction_dependent" in terminal["reason"]
