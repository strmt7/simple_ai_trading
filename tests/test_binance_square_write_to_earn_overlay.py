from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-square-organic-write-to-earn-fee-overlay-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "29ec95146998535fde295dfc830a2639b9d10964e7f9e36c17e44e628dc454d1"
REGISTRY_HASH = "8a5df5625fab7d55762ff52923f1454d80a92126d6dce09ce4f5b9281779f6f9"


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


def test_write_to_earn_overlay_is_hash_bound_scoped_and_non_trading() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["adjudication"]["accepted_edge"] is True
    assert artifact["adjudication"]["market_direction_forecast_required"] is False
    assert artifact["adjudication"]["deployment_ready"] is False
    assert artifact["adjudication"]["profitability_claim"] is False
    assert artifact["authority"] == {
        "account_state_accessed": False,
        "content_published": 0,
        "credentials_used": False,
        "funded_actions": 0,
        "orders_or_transfers_submitted": 0,
        "public_read_only_research": True,
        "reader_activity_created_or_requested": 0,
    }


def test_only_base_commission_is_accepted_with_threshold_and_integrity_guards() -> None:
    artifact = _load(ARTIFACT)
    economics = artifact["economics"]
    terms = artifact["eligibility_and_terms"]

    assert Decimal(economics["accepted_base_commission_rate_of_reader_trading_fee"]) == (
        Decimal("0.20")
    )
    assert Decimal(1) * Decimal(
        economics["accepted_base_commission_rate_of_reader_trading_fee"]
    ) == Decimal(economics["accepted_gross_USDC_per_1_USDC_equivalent_eligible_reader_fee"])
    assert economics["minimum_weekly_payout_USDC"] == "0.1"
    assert economics["sub_threshold_balance_carry_forward"] is False
    assert terms["content_reward_lifetime_days"] == 7
    assert terms["self_trades_eligible"] is False
    assert terms["zero_fee_trades_eligible"] is False
    prohibited = " ".join(artifact["prohibited_actions"])
    assert "manufactured reader trading" in prohibited
    assert "encouraging unnecessary unsuitable leveraged or loss-making trades" in prohibited
    assert "double-counting the same reader fee" in prohibited


def test_registry_accepts_overlay_without_crediting_unproved_bonus_rates() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 43)
    )
    row = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_square_organic_write_to_earn_fee_overlay"
    )
    assert row["priority_rank"] == 27
    assert row["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-square-organic-write-to-earn-fee-overlay-v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        }
    ]
    assert "base_20_percent" in row["current_status"]
    assert "50_percent" not in row["current_status"]
