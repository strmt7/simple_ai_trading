from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs/model-research/action-value/"
    "polymarket-perps-binance-bfusd-collateral-stack-gate-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "a6ff387d70d33c40951e36de93eff7c810b2291dbefff5ecb0f3953880fe7878"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_live_bfusd_terms_reconstruct_and_reject_before_account_access() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact) == EXPECTED_HASH
    public = artifact["calculation"]["current_public_bfusd_product_page"]
    assert public["seven_day_average_apr_percent"] == "5.03"
    assert public["last_day_apr_percent"] == "5.12"
    assert public["purchase_fee_percent"] == "0"
    assert public["fast_redemption"]["fee_free_first_bfusd"] == "500"
    assert public["fast_redemption"]["fee_rate_percent_after_free_tranche"] == (
        "0.1"
    )
    assert public["standard_redemption"]["fee_rate_percent"] == "0.025"

    threshold = artifact["calculation"]["role_persistence"]
    assert threshold["worst_role_existing_collateral_annual_percent"] == (
        "14.10661947368421052631578947331578947368"
    )
    assert artifact["decision"]["account_evidence_escalation_permitted_now"] is False
    assert artifact["decision"]["accepted_edge"] is False
    assert artifact["preflight"]["signed_requests"] == 0
    assert artifact["preflight"]["venue_market_or_account_http_requests"] == 0


def test_registry_points_to_current_bfusd_gate_without_promotion() -> None:
    registry = _load(REGISTRY)
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "conditional_polymarket_perps_OI_reward_and_funding_hedged_by_binance_perpetual_with_BFUSD_collateral_rewards"
    )

    assert candidate["canonical_artifacts"][-1]["result_sha256"] == EXPECTED_HASH
    assert candidate["priority_rank"] == 20
    assert registry["accepted_edge_count"] == 13
    assert registry["authority"]["profitability_claim"] is False
