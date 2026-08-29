from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / (
    "binance-blvt-primary-market-nav-parity-contract-v1-2026-08-29.json"
)
RESULT = ACTION_VALUE / (
    "binance-blvt-primary-market-nav-parity-public-gate-v1-2026-08-29.json"
)
RUNNER = ROOT / "tools/capture_binance_blvt_public_inventory.py"
DATA_ROOT = ROOT / "data/binance-blvt-public-inventory-v1"
RAW = DATA_ROOT / "raw/exchange-info.json"
JOURNAL = DATA_ROOT / "request-journal.jsonl"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "9189fd384c2185875a8682155f365eb79cfe5f1061016ca035b05dfc88545b9f"
RESULT_HASH = "85c8ef364b03fb2fbf0aeebddec10d51abbdd608f56ff9c0dccb1835cacc2179"
REGISTRY_HASH = "8a5df5625fab7d55762ff52923f1454d80a92126d6dce09ce4f5b9281779f6f9"


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


def test_contract_freezes_one_public_inventory_request_and_no_actions() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert contract["capture"]["request_count"] == 1
    assert contract["capture"]["retry_count"] == 0
    assert contract["authority"] == {
        **contract["authority"],
        "credentials_used": False,
        "account_requests": 0,
        "tokenInfo_requests": 0,
        "spot_or_futures_book_requests": 0,
        "subscriptions_or_redemptions": 0,
        "orders_placed": 0,
        "funds_used": False,
        "trading_authority": False,
    }
    assert (
        contract["implementation"]["sha256"]
        == hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    )


def test_public_inventory_has_no_current_trading_blvt() -> None:
    result = _load(RESULT)
    receipts = [json.loads(line) for line in JOURNAL.read_text().splitlines()]

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["contract"]["sha256"] == CONTRACT_HASH
    inventory = result["public_inventory"]
    assert inventory["exchange_info_symbol_count"] == 3685
    assert inventory["leveraged_permission_symbol_count"] == 40
    assert inventory["trading_leveraged_symbol_count"] == 0
    assert inventory["trading_leveraged_symbols"] == []
    assert inventory["raw_sha256"] == hashlib.sha256(RAW.read_bytes()).hexdigest()
    assert len(RAW.read_bytes()) == 17_532_893
    assert len(receipts) == 2
    assert receipts[0]["phase"] == "intent"
    assert receipts[1]["phase"] == "completed"
    assert receipts[1]["status_code"] == 200
    assert receipts[1]["response_sha256"] == inventory["raw_sha256"]


def test_zero_inventory_terminalizes_only_current_population() -> None:
    result = _load(RESULT)

    assert result["adjudication"] == {
        "accepted_edge": False,
        "candidate_edge": False,
        "deployment_ready": False,
        "next_action": "terminalize_the_current_public_inventory_until_a_new_BLVT_listing",
        "profitability_claim": False,
    }
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
    assert result["authority"]["signed_requests"] == 0
    assert result["authority"]["orders_placed"] == 0


def test_registry_records_terminal_blvt_mechanism_without_acceptance() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 43)
    )
    row = registry["prioritized_hypotheses"][-1]
    assert row["priority_rank"] == 42
    assert row["mechanism"] == (
        "binance_BLVT_primary_market_NAV_subscription_redemption_spot_parity_with_"
        "exact_basket_hedge"
    )
    assert [item["result_sha256"] for item in row["canonical_artifacts"]] == [
        CONTRACT_HASH,
        RESULT_HASH,
    ]
    assert "zero_with_TRADING_status" in row["current_status"]
    assert "new_official_BLVT_spot_listing_or_relisting" in row["next_action"]
