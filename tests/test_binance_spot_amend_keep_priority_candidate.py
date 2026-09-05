from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-spot-amend-keep-priority-source-contract-v1-2026-08-30.json"
)
ADJUDICATION = ROOT / (
    "docs/model-research/action-value/"
    "binance-spot-amend-keep-priority-source-failure-adjudication-v1-2026-08-30.json"
)
JOURNAL = ROOT / "data/binance-spot-amend-keep-priority-source-v1/journal.json"
RAW = (
    ROOT / "data/binance-spot-amend-keep-priority-source-v1/raw/"
    "order_amend_keep_priority.raw.md"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical(body))


def _normalize_markdown_for_frozen_offline_adjudication(text: str) -> str:
    return text.replace("**", "").replace("`", "").replace("<br>", "")


def test_consumed_source_failure_and_offline_semantics_reconstruct() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    adjudication = json.loads(ADJUDICATION.read_text(encoding="ascii"))
    journal = json.loads(JOURNAL.read_text(encoding="ascii"))

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    implementation = contract["implementation"]
    assert (
        _sha256((ROOT / implementation["path"]).read_bytes())
        == implementation["sha256"]
    )
    lineage = adjudication["consumed_run"]["retained_source_line_ending_reconstruction"]
    for source in contract["retained_sources"]:
        payload = (ROOT / source["path"]).read_bytes()
        if source["path"] == lineage["path"] and _sha256(payload) != source["sha256"]:
            assert _sha256(payload) == lineage["durable_git_blob_sha256"]
            assert _sha256(payload.replace(b"\n", b"\r\n")) == source["sha256"]
            assert source["sha256"] == lineage["frozen_worktree_sha256"]
        else:
            assert _sha256(payload) == source["sha256"]
    native_index = (ROOT / contract["retained_sources"][0]["path"]).read_text(
        encoding="utf-8"
    )
    assert (
        "`PUT /api/v3/order/amend/keepPriority` — Order Amend Keep Priority (TRADE)"
        in native_index
    )
    changelog = (ROOT / contract["retained_sources"][1]["path"]).read_text(
        encoding="utf-8"
    )
    assert "request weights have been increased from 1 to 4" in changelog
    assert "Order Amend Keep Priority will be enabled on all symbols" in changelog

    assert journal["state"] == "failed"
    assert journal["contract_sha256"] == contract["contract_sha256"]
    assert len(journal["requests"]) == 1
    request = journal["requests"][0]
    raw = RAW.read_bytes()
    assert request["method"] == "GET"
    assert request["status_code"] == 200
    assert request["response_bytes"] == len(raw) == 4705
    assert request["response_sha256"] == _sha256(raw)
    assert request["request_body_sha256"] == _sha256(b"")
    assert journal["error_type"] == "ValueError"

    normalized = _normalize_markdown_for_frozen_offline_adjudication(
        raw.decode("utf-8").casefold()
    )
    for phrase in contract["public_source"]["required_text_casefolded"]:
        assert phrase.casefold().replace("`", "") in normalized
    facts = adjudication["offline_retained_source_facts"]
    assert facts["successful_amend_keeps_time_priority_at_the_same_price"] is True
    assert (
        facts[
            "cancel_replace_loses_time_priority_and_executes_after_existing_same_price_orders"
        ]
        is True
    )
    assert facts["failed_amend_is_rejected_and_leaves_the_order_unchanged"] is True
    assert facts["unfilled_order_count_per_amend"] == 0
    assert facts["request_weight"] == 4
    assert _self_hash(adjudication, "result_sha256") == adjudication["result_sha256"]


def test_retained_production_configuration_and_rank_five_lineage_reconstruct() -> None:
    adjudication = json.loads(ADJUDICATION.read_text(encoding="ascii"))
    config = adjudication["retained_production_configuration"]
    compressed = (ROOT / config["source_path"]).read_bytes()
    decompressed = gzip.decompress(compressed)
    assert _sha256(compressed) == config["gzip_sha256"]
    assert _sha256(decompressed) == config["decompressed_sha256"]
    payload = json.loads(decompressed)
    assert len(payload["symbols"]) == config["complete_symbol_count"] == 3685
    for expected in config["scoped_rows"]:
        row = next(
            item for item in payload["symbols"] if item["symbol"] == expected["symbol"]
        )
        amend_filter = next(
            item
            for item in row["filters"]
            if item["filterType"] == "MAX_NUM_ORDER_AMENDS"
        )
        assert row["status"] == expected["status"] == "TRADING"
        assert row["amendAllowed"] is expected["amendAllowed"] is True
        assert amend_filter["maxNumOrderAmends"] == expected["maxNumOrderAmends"] == 10

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_five = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert {
        "path": ADJUDICATION.relative_to(ROOT).as_posix(),
        "result_sha256": adjudication["result_sha256"],
    } in rank_five["canonical_artifacts"]
    assert adjudication["adjudication"]["accepted_edge"] is False
    assert (
        adjudication["economic_adjudication"]["public_forward_profit_floor_quote_units"]
        == "0"
    )


def test_markdown_gate_rule_records_the_consumed_methodology_correction() -> None:
    agents = (ROOT / "docs" / "RESEARCH_CAPTURE_BOUNDARIES.md").read_text(
        encoding="utf-8"
    )
    assert "Freeze text gates against the exact retained representation" in agents
    assert "adjudicate the retained bytes offline" in agents
    assert "never refetch" in agents
