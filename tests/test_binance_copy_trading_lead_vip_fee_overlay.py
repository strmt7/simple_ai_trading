from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ACTION / "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
EDGE = ACTION / "binance-copy-trading-lead-vip-fee-overlay-edge-v1-2026-09-01.json"
ANNOUNCEMENT_CONTRACT = (
    ACTION
    / "binance-copy-trading-lead-vip-fee-overlay-source-contract-v1-2026-09-01.json"
)
ANNOUNCEMENT_RESULT = (
    ACTION
    / "binance-copy-trading-lead-vip-fee-overlay-source-result-v1-2026-09-01.json"
)
FAQ_CONTRACT = (
    ACTION
    / "binance-copy-trading-lead-vip-fee-faq-source-contract-v1-2026-09-01.json"
)
FAQ_RESULT = (
    ACTION
    / "binance-copy-trading-lead-vip-fee-faq-source-result-v1-2026-09-01.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _body_text(raw: dict[str, object]) -> str:
    tree = json.loads(raw["data"]["body"])
    values: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("node") == "text":
                values.append(str(node["text"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return " ".join(values)


def test_copy_trading_lead_vip_sources_and_terms_are_bound() -> None:
    edge = _load(EDGE)
    for contract_path, result_path, source_key in (
        (ANNOUNCEMENT_CONTRACT, ANNOUNCEMENT_RESULT, "announcement"),
        (FAQ_CONTRACT, FAQ_RESULT, "program_faq"),
    ):
        contract = _load(contract_path)
        result = _load(result_path)
        source = edge["source_binding"][source_key]
        raw = ROOT / source["raw_path"]

        assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
        assert _self_hash(result, "result_sha256") == result["result_sha256"]
        assert result["contract"]["sha256"] == contract["contract_sha256"]
        assert _sha256(raw) == source["raw_sha256"]
        assert result["source_gate"]["passed"] is True

    faq = _load(ROOT / edge["source_binding"]["program_faq"]["raw_path"])
    text = _body_text(faq)
    for term in (
        "≥ 200 copiers",
        "≥ 50 copiers",
        "≥ 5,000,000 USDT",
        "Upgrade to V6 or +2",
        "at least 25 BNB",
        "1st week of each month",
    ):
        assert term in text


def test_copy_trading_lead_vip_edge_is_scoped_and_fail_closed() -> None:
    edge = _load(EDGE)
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert _self_hash(edge, "result_sha256") == edge["result_sha256"]
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _self_hash(audit, "result_sha256") == audit["result_sha256"]
    assert edge["adjudication"]["accepted_edge"] is True
    assert edge["adjudication"]["deployment_ready"] is False
    assert edge["adjudication"]["public_forward_net_saving_floor_USD"] == "0"
    assert edge["registry_effect"]["accepted_edge_count"] == 30
    assert edge["retained_fee_sensitivity"][
        "minimum_displayed_positive_example_at_5000000_USDT_trailing_volume_USD"
    ] == "90"

    rank_five = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert {
        "path": str(EDGE.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": edge["result_sha256"],
    } in rank_five["canonical_artifacts"]
    assert "authentic_organic_lead_flow" in rank_five[
        "copy_trading_lead_vip_current_status"
    ]
    assert registry["accepted_edge_count"] == 30
    assert audit["source_binding"]["accepted_edge_count"] == 30
    external = next(
        group
        for group in audit["classification"]
        if group["evidence_tier"] == "D_external_user_or_client_revenue_overlay"
    )
    assert external["count"] == 9
    assert 30 in external["edge_ordinals"]

    prohibited = " ".join(edge["prohibited_shortcuts"])
    assert "fake copiers" in prohibited
    assert "testnet credentials" in prohibited
    assert edge["authority"]["credentials_used"] is False
    assert edge["authority"]["orders_or_transactions"] == 0
