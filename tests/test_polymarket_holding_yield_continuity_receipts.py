from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools import reconcile_polymarket_holding_yield_continuity_receipts as monitor


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-continuity-receipts-contract-v8-2026-08-29.json"
)
RESULT = ROOT / (
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-continuity-receipts-v8-2026-08-29.json"
)
JOURNAL = ROOT / "data/polymarket-holding-yield-continuity-receipts-v8/journal.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "ace38fca480049601d876fb8ae781b5103372662e9294e78c5905d8162332b42"
RESULT_HASH = "2eb7b434170afb195cc4f4faef8260ac4ec30b655c20fc07ee1bc9acbdfe090d"
REGISTRY_HASH = "0a34d7289331515f8e7b3f09e856fbc331ecbc3a91130fea20542a39ef211f60"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_contract_and_retained_source_lineage_reconstruct() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    implementation = contract["implementation"]
    assert (
        hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
        == (implementation["sha256"])
    )
    for case in contract["cases"]:
        validated = monitor._validate_case(case)
        assert validated["retained_pair_equal_and_mergeable"] is True
        assert validated["only_selected_yield_in_interval"] is True
        for source_name in ("activity_source", "positions_source"):
            source = case[source_name]
            assert (
                hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
                == (source["sha256"])
            )


def test_result_and_exact_distributor_transfers_reconstruct() -> None:
    result = _load(RESULT)

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert [row["selected_interval_seconds"] for row in result["cases"]] == [
        86474,
        86724,
    ]
    assert [row["selected_amount_pusd"] for row in result["cases"]] == [
        "0.0133",
        "0.0391",
    ]
    for case in result["cases"]:
        receipt = case["receipt"]
        raw_path = ROOT / receipt["source"]["raw_path"]
        payload = raw_path.read_bytes()
        assert (
            hashlib.sha256(payload).hexdigest() == receipt["source"]["response_sha256"]
        )
        envelope = json.loads(payload)
        assert monitor._payout_transfer(
            envelope["result"],
            wallet=case["wallet"],
            amount=Decimal(case["selected_amount_pusd"]),
        )
    assert result["adjudication"] == {
        "accepted_historical_scoped_edge_preserved": True,
        "all_latest_retained_receipts_reconciled": True,
        "current_rate_qualified_by_this_monitor": False,
        "deployment_ready": False,
        "future_profit_guaranteed": False,
        "next_action": (
            "continue_only_distinct_public_continuity_monitoring_or_wait_for_"
            "material_terms_or_economics_change"
        ),
        "public_profit_floor_for_new_capital_pusd": "0",
        "status": "two_additional_daily_payout_receipts_reconciled",
    }


def test_journal_precommits_exact_two_request_bodies() -> None:
    contract = _load(CONTRACT)
    journal = _load(JOURNAL)

    assert journal["state"] == "completed"
    assert journal["contract_sha256"] == CONTRACT_HASH
    assert journal["result_sha256"] == RESULT_HASH
    assert len(journal["requests"]) == 2
    for request_id, (case, request) in enumerate(
        zip(contract["cases"], journal["requests"], strict=True)
    ):
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "eth_getTransactionReceipt",
            "params": [case["selected_transaction_hash"]],
        }
        assert (
            request["body_sha256"]
            == hashlib.sha256(monitor._canonical_json(body).encode("ascii")).hexdigest()
        )
        assert request["state"] == "received"
        assert request["status_code"] == 200


def test_registry_preserves_scope_and_adds_continuity_artifact() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 21
    assert len(registry["prioritized_hypotheses"]) == 44
    row = registry["prioritized_hypotheses"][0]
    assert row["mechanism"] == "complete_set_holding_reward"
    assert row["canonical_artifacts"][-1] == {
        "path": (
            "docs/model-research/polymarket/"
            "complete-set-holding-yield-continuity-receipts-v8-2026-08-29.json"
        ),
        "result_sha256": RESULT_HASH,
    }
    assert "current_rate_remains_fail_closed_unqualified" in row["current_status"]
