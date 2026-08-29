from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs/model-research/action-value"
    / "binance-vip-earn-idle-inventory-overlay-candidate-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
ARTIFACT_HASH = "6c83a950c856202c511b7a7717be3e154cfe8aeed78b84bc89378c7d017ec692"
PUBLIC_TERMINAL_HASH = "cd41cad8e0053b9d41ddda64fd4ad8a86a163307ddcc9fabc805c56b9c5028c9"
REGISTRY_HASH = "c0d7189c4848f248e6d3960954198e0f1e93c8e74acd2ed36a8830239bf86194"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_vip_earn_candidate_is_hash_bound_and_action_free() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == ARTIFACT_HASH
    assert _canonical_hash(artifact, "result_sha256") == ARTIFACT_HASH
    assert artifact["authority"]["authenticated_requests"] == 0
    assert artifact["authority"]["subscriptions_redemptions_borrows_transfers_orders_or_quotes"] == 0
    assert artifact["current_evidence_boundary"]["public_forward_incremental_profit_floor"] == "0"
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "profitability_claim": False,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "status": "materially_new_direction_independent_incremental_yield_candidate_only_for_independently_existing_VIP_status_and_idle_eligible_inventory_exact_products_rates_quotas_account_eligibility_and_after_cost_value_unproved",
    }
    assert all(source["status"] == 200 for source in artifact["sources"])


def test_registry_adds_only_the_unaccepted_conditional_overlay() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 21
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 45))
    hypothesis = next(
        row
        for row in hypotheses
        if row["mechanism"]
        == "binance_VIP_Earn_existing_idle_inventory_incremental_yield_overlay"
    )
    assert hypothesis["priority_rank"] == 39
    assert hypothesis["canonical_artifacts"] == [
        {
            "path": "docs/model-research/action-value/binance-vip-earn-idle-inventory-overlay-candidate-v1-2026-08-27.json",
            "result_sha256": ARTIFACT_HASH,
        },
        {
            "path": "docs/model-research/action-value/binance-vip-earn-public-btc-eth-sol-comparator-terminal-v1-2026-08-27.json",
            "result_sha256": PUBLIC_TERMINAL_HASH,
        },
    ]
