from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "binance-copy-trading-lead-profit-share-rendered-source-contract-"
    "v1-2026-08-30.json"
)
EVIDENCE = BASE / (
    "binance-copy-trading-lead-profit-share-rendered-evidence-"
    "v1-2026-08-30.json"
)
ARTIFACT = BASE / (
    "binance-copy-trading-realized-organic-follower-profit-share-"
    "overlay-v1-2026-08-30.json"
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


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return hashlib.sha256(_canonical(body)).hexdigest()


def test_copy_trading_rendered_source_contract_reconstructs() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    evidence = json.loads(EVIDENCE.read_text(encoding="ascii"))

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert (
        _self_hash(evidence, "rendered_evidence_sha256")
        == evidence["rendered_evidence_sha256"]
    )
    assert evidence["contract"]["contract_sha256"] == contract["contract_sha256"]
    assert evidence["rendered_source"]["total_rendered_lines"] == 348
    assert evidence["admission"]["exact_current_product_semantics_admitted"] is True
    assert evidence["admission"][
        "account_eligibility_rate_followers_or_owned_income_proved"
    ] is False
    assert all(row["passed"] for row in evidence["requirement_results"].values())


def test_realized_profit_share_overlay_is_narrow_and_fail_closed() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _self_hash(artifact, "result_sha256") == artifact["result_sha256"]
    decision = artifact["adjudication"]
    authority = artifact["authority"]
    assert decision["accepted_edge"] is True
    assert decision["accepted_edge_count_after"] == (
        decision["accepted_edge_count_before"] + 1
    )
    assert decision["market_direction_forecast_required_for_incremental_profit_share"] is False
    assert decision["profitability_claim"] is False
    assert decision["deployment_ready"] is False
    assert decision["public_forward_profit_floor"] == "0"
    assert authority["credentials_used"] is False
    assert authority["signed_requests"] == 0
    assert authority["orders_or_trades"] == 0
    assert authority["state_changes"] == 0
    assert artifact["economic_contract"]["rate_assumption"] == "none"
    assert artifact["account_reconciliation"]["state_changes_authorized"] is False

    source = artifact["source_binding"]["retained_current_official_api_index"]
    retained = (ROOT / source["path"]).read_bytes()
    assert hashlib.sha256(retained).hexdigest() == source["sha256"]
    decoded = retained.decode("utf-8")
    for fragment in source["required_fragments"]:
        assert fragment in decoded


def test_copy_trading_overlay_is_registered_without_accepting_strategy_alpha() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "organic_third_party_platform_fee_overlays"
    )
    assert family["priority_rank"] == 24
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "independently_cross_regime_accepted" in family[
        "accepted_copy_trading_profit_share_scope"
    ]
    assert "classifies_it_TRADE" in family[
        "accepted_copy_trading_profit_share_next_action"
    ]
