from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/model-research/action-value/"
    "binance-spot-fix-unordered-execution-risk-candidate-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fix_source_and_candidate_lineage_are_hash_bound() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]

    sources = result["sources"]
    contract_source = sources["contract"]
    contract_path = ROOT / contract_source["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert _sha256(contract_path) == contract_source["file_sha256"]
    assert _canonical_hash(contract, "contract_sha256") == contract_source[
        "contract_sha256"
    ]

    source_result_source = sources["source_result"]
    source_result_path = ROOT / source_result_source["path"]
    source_result = json.loads(source_result_path.read_bytes())
    assert _sha256(source_result_path) == source_result_source["file_sha256"]
    assert _canonical_hash(source_result, "result_sha256") == source_result_source[
        "result_sha256"
    ]
    assert source_result["source_gate"]["passed"] is True

    for source_name in ("retained_FIX_excerpt", "request_journal"):
        source = sources[source_name]
        assert _sha256(ROOT / source["path"]) == source["sha256"]

    receipt = sources["source_receipt"]
    assert receipt["response_sha256"] == source_result["capture"]["receipt"][
        "response_sha256"
    ]
    assert receipt["response_bytes"] == source_result["capture"]["receipt"][
        "response_bytes"
    ]
    manifest_source = sources["secret_free_extraction_manifest"]
    manifest_path = ROOT / manifest_source["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert _sha256(manifest_path) == manifest_source["file_sha256"]
    assert _canonical_hash(manifest, "manifest_sha256") == manifest_source[
        "manifest_sha256"
    ]
    assert manifest["extraction"]["private_key_block_count"] == 0
    assert manifest["disposition"]["full_payload_committed"] is False


def test_fix_mechanism_does_not_invent_cancel_on_disconnect_or_profit() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    terms = result["retained_current_terms"]
    hypothesis = result["hypothesis_adjudication"]
    source = result["sources"]["retained_FIX_excerpt"]
    text = (ROOT / source["path"]).read_text(encoding="utf-8")

    assert "`UNORDERED(1)` should offer better performance" in text
    assert "FIX API should give better performance" in text
    assert "All orders of the account will be canceled" in text
    assert "cancel on disconnect" not in text.lower()
    assert "BEGIN PRIVATE KEY" not in text
    assert terms["maximum_concurrent_order_entry_connections_per_account"] == 10
    assert terms["unfilled_order_count_scope"] == "account_not_connection"
    assert terms["drop_copy_delay_seconds"] == 1
    assert terms["testnet_or_demo_FIX_order_entry_support_in_retained_source"] is False
    assert hypothesis["automatic_cancel_on_disconnect_documented"] is False
    assert hypothesis["accepted_edge"] is False
    assert hypothesis["standalone_after_cost_profit_floor_USDT"] == "0"

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_5 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert rank_5["fix_unordered_execution_retry_trigger"] == result["adjudication"][
        "retry_trigger"
    ]
    assert rank_5["fix_unordered_next_action"] == result["adjudication"][
        "next_action"
    ]
    assert any(
        row["result_sha256"] == result["result_sha256"]
        for row in rank_5["canonical_artifacts"]
    )
    assert any(
        row["canonical_result_sha256"] == result["result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    )
