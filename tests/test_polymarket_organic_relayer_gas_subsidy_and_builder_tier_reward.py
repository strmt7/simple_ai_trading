import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-organic-builder-tier-reward-source-contract-v1.json"
)
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-organic-relayer-gas-subsidy-and-builder-tier-reward-"
    "v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(document: dict, field: str) -> str:
    payload = dict(document)
    expected = payload.pop(field)
    actual = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert actual == expected
    return actual


def test_builder_tier_source_is_exact_hash_bound_and_one_use() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    _canonical_hash(contract, "contract_sha256")
    _canonical_hash(result, "result_sha256")
    assert result["contract"]["contract_sha256"] == contract["contract_sha256"]

    source = result["source"]
    raw = ROOT / source["raw_path"]
    journal = ROOT / source["journal_path"]
    raw_bytes = raw.read_bytes()
    assert len(raw_bytes) == source["raw_bytes"] == 6200
    assert hashlib.sha256(raw_bytes).hexdigest() == source["raw_sha256"]
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == source["journal_sha256"]

    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [entry["phase"] for entry in entries] == ["intent", "completed"]
    assert entries[0]["requested_before_utc"] == entries[1]["requested_before_utc"]
    assert entries[1]["status_code"] == 200
    assert entries[1]["response_sha256"] == source["raw_sha256"]
    assert source["network_requests_used"] == 1
    assert contract["authority"]["venue_market_data_requests_permitted"] == 0


def test_official_source_proves_gas_subsidy_but_not_cash_reward_formula() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    source_text = (ROOT / result["source"]["raw_path"]).read_text(encoding="utf-8")

    assert "Gas fees subsidized for supported smart-wallet operations" in source_text
    assert "Gas subsidized on all Relayer transactions up to the daily limit" in source_text
    assert "Weekly USDC rewards based on volume (subject to approval)" in source_text
    assert "Grants (subject to approval)" in source_text

    terms = result["official_tier_terms"]
    assert terms["unverified"]["daily_relayer_transaction_limit"] == 100
    assert terms["verified"]["daily_relayer_transaction_limit"] == 10000
    assert terms["partner"]["daily_relayer_transaction_limit"] == "unlimited"

    subsidy = result["gas_subsidy_candidate"]
    assert subsidy["accepted_edge"] is False
    assert subsidy["accepted_edge_count_change"] == 0
    assert subsidy["market_direction_forecast_required"] is False
    assert subsidy["standalone_profitability_claim"] is False
    assert subsidy["deployment_ready"] is False
    assert subsidy["public_forward_profit_floor_pUSD"] == "0"
    assert subsidy["zero_activity_floor_pUSD"] == "0"

    rewards = result["unaccepted_tier_cash_rewards"]
    assert rewards["accepted_edge"] is False
    assert rewards["public_forward_floor_USDC"] == "0"


def test_registry_routes_relayer_overlay_without_new_family() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    _canonical_hash(registry, "result_sha256")
    assert registry["accepted_edge_count"] == 21
    assert len(registry["prioritized_hypotheses"]) == 44

    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 24
    )
    assert result["result_sha256"] in {
        artifact["result_sha256"] for artifact in family["canonical_artifacts"]
    }
    assert "Relayer" in family["current_status"]
    assert "gas_subsidy" in family["retry_trigger"]
    assert any(
        "creating_splitting_retrying_or_rerouting_transactions" in shortcut
        for shortcut in family["prohibited_shortcuts"]
    )
