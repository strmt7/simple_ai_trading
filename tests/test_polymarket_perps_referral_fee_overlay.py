from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "polymarket-perps-organic-referral-fee-overlay-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "4bebea610dc9406d598627035f4e6e815e6a4daeb64944d7ba2ec9f55b6b7d71"
REGISTRY_HASH = "c0d7189c4848f248e6d3960954198e0f1e93c8e74acd2ed36a8830239bf86194"


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


def test_perps_referral_overlay_is_hash_bound_scoped_and_non_trading() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["adjudication"]["accepted_edge"] is True
    assert artifact["adjudication"]["market_direction_forecast_required"] is False
    assert artifact["adjudication"]["deployment_ready"] is False
    assert artifact["adjudication"]["profitability_claim"] is False
    assert artifact["authority"] == {
        "account_state_accessed": False,
        "authenticated_requests": 0,
        "credentials_used": False,
        "funded_actions": 0,
        "orders_or_transfers_submitted": 0,
        "public_read_only_research": True,
        "referral_links_created_or_shared": 0,
        "referred_activity_created_or_requested": 0,
    }


def test_fee_share_and_invite_limits_reject_volume_farming() -> None:
    artifact = _load(ARTIFACT)
    economics = artifact["economics"]
    terms = artifact["eligibility_and_terms"]

    assert Decimal(economics["gross_fee_share_rate"]) == Decimal("0.20")
    assert economics["per_referred_trader_reward_cap"] is None
    assert economics["payout_timing"] == "weekly"
    assert economics["payout_asset_publicly_unproved"] is True
    assert terms["initial_available_invites"] == 10
    assert terms["standard_limit_after_initial_active_referrals"] == 25
    assert terms["self_code_application_allowed"] is False
    assert terms["prediction_market_referral_program_separate"] is True
    prohibited = " ".join(artifact["prohibited_actions"])
    assert "own Perps volume" in prohibited
    assert "unnecessary unsuitable leveraged or loss-making Perps trades" in prohibited
    assert "available-invite count is exhausted" in prohibited


def test_registry_accepts_perps_referral_as_separate_available_invite_overlay() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 21
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 45)
    )
    row = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "polymarket_Perps_organic_referral_fee_overlay"
    )
    assert row["priority_rank"] == 29
    assert row["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-perps-organic-referral-fee-overlay-v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        }
    ]
    assert "account_confirmed_available_invites" in row["current_status"]
