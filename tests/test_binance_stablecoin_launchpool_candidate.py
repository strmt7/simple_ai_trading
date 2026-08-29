from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-stablecoin-launchpool-idle-inventory-reward-candidate-v1-"
    "2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "f898914a56fe61c063ca0eaf8d02fc91ea8bf527dd3ff49289527db524d286c3"
REGISTRY_HASH = "fad104fc4e460bd0ff69e5b61df95050bbc0570d0af2f3eff64f8403b00b61bb"


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


def test_stablecoin_launchpool_is_hash_bound_unaccepted_and_non_mutating() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "status": (
            "candidate_direction_independent_incremental_launchpool_reward_on_"
            "already_held_idle_supported_stablecoin_no_current_active_offer_"
            "account_eligibility_or_after_cost_reward_value_proved"
        ),
        "trading_authority": False,
    }
    assert artifact["authority"]["authenticated_requests"] == 0
    assert artifact["authority"]["launchpool_locks_or_unlocks_submitted"] == 0
    assert artifact["authority"]["orders_or_transfers_submitted"] == 0


def test_candidate_waits_for_current_offer_and_values_only_owned_sale_proceeds() -> None:
    artifact = _load(ARTIFACT)
    current = artifact["current_offer_state"]
    example = artifact["historical_mechanism_example"]

    assert current["direct_current_launchpool_page_status"] == 202
    assert current["direct_current_launchpool_page_waf_empty"] is True
    assert current["active_project_proved"] is False
    assert current["public_forward_reward_floor"] == "0"
    assert example["historical_only"] is True
    assert example["farming_duration_days"] == 2
    assert example["farming_end_exact_publicly_unproved"] is True
    assert {"USDC", "U", "USD1"}.issubset(example["supported_assets"])
    valuation = artifact["economic_gate"]["reward_valuation_rule"]
    assert "owned distributed reward quantity" in valuation
    assert "executable displayed bids" in valuation


def test_registry_adds_trigger_based_stablecoin_overlay_without_acceptance() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 42)
    )
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "binance_stablecoin_launchpool_idle_inventory_reward_overlay"
    )
    assert row["priority_rank"] == 32
    assert row["market_direction_forecast_required"] is False
    assert row["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-stablecoin-launchpool-idle-inventory-reward-candidate-"
                "v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        }
    ]
    assert "WAF_empty" in row["current_status"]
