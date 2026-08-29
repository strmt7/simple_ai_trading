from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-organic-referral-pro-fee-overlay-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "8a29116879fd90cb0f8fc11d9780a8dccbff8afc2d3ea685e671921f651e64d1"
REGISTRY_HASH = "7be2ae9f2883a72c6f492e57208cede33c7ad8733a947bf4a21b3825315b2443"


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


def test_referral_pro_overlay_is_hash_bound_scoped_and_non_trading() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["adjudication"]["accepted_edge"] is True
    assert artifact["adjudication"]["market_direction_forecast_required"] is False
    assert artifact["adjudication"]["deployment_ready"] is False
    assert artifact["adjudication"]["profitability_claim"] is False
    assert artifact["authority"] == {
        "account_state_accessed": False,
        "credentials_used": False,
        "funded_actions": 0,
        "orders_or_transfers_submitted": 0,
        "public_read_only_research": True,
        "referral_links_created_or_shared": 0,
        "referred_activity_created_or_requested": 0,
    }


def test_only_public_base_rates_are_accepted_with_mode_and_integrity_guards() -> None:
    artifact = _load(ARTIFACT)
    economics = artifact["economics"]
    terms = artifact["eligibility_and_terms"]

    assert Decimal(economics["accepted_spot_and_margin_base_commission_rate"]) == (
        Decimal("0.20")
    )
    assert Decimal(economics["accepted_futures_base_commission_rate"]) == Decimal(
        "0.10"
    )
    assert economics["accepted_futures_base_commission_time_limit"].startswith(
        "1 year"
    )
    assert economics["payout_asset_and_timing_publicly_unproved"] is True
    assert "only one" in terms["new_user_single_mode"]
    assert terms["restricted_region_users_eligible"] is False
    prohibited = " ".join(artifact["prohibited_actions"])
    assert "inauthentic referral" in prohibited
    assert "unnecessary unsuitable leveraged or loss-making trades" in prohibited
    assert "double-counting the same user or fee" in prohibited
    assert "assuming the commission payout asset" in prohibited


def test_registry_accepts_referral_pro_without_crediting_performance_tiers() -> None:
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
        if row["mechanism"] == "binance_organic_referral_pro_fee_overlay"
    )
    assert row["priority_rank"] == 28
    assert row["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-organic-referral-pro-fee-overlay-v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        }
    ]
    assert "base_20_percent_Spot_and_Margin" in row["current_status"]
    assert "base_10_percent_Futures" in row["current_status"]
