from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "binance-link-trade-existing-client-kickback-overlay-"
    "rendered-source-contract-v1-2026-08-30.json"
)
EVIDENCE = BASE / (
    "binance-link-trade-existing-client-kickback-overlay-"
    "rendered-evidence-v1-2026-08-30.json"
)
ARTIFACT = BASE / (
    "binance-link-trade-existing-client-kickback-overlay-"
    "edge-v1-2026-08-30.json"
)
PARTNER_ARTIFACT = BASE / (
    "binance-link-trade-partner-rebate-realized-organic-client-"
    "overlay-edge-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
FRONTIER = BASE / "accepted-market-independent-yield-frontier-v1-2026-08-30.json"


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


def test_rendered_contract_and_extraction_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    evidence = json.loads(EVIDENCE.read_text(encoding="ascii"))

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert (
        _self_hash(evidence, "rendered_evidence_sha256")
        == evidence["rendered_evidence_sha256"]
    )
    assert evidence["contract"]["contract_sha256"] == contract["contract_sha256"]
    assert evidence["rendered_source"]["total_rendered_lines"] == 747
    assert evidence["admission"]["exact_current_api_contract_admitted"] is True
    assert evidence["admission"]["documentation_example_values_admitted"] is False
    assert evidence["admission"]["owned_income_proved"] is False
    for result in evidence["requirement_results"].values():
        assert result["passed"] is True


def test_official_index_keeps_client_and_partner_records_distinct() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    source = artifact["retained_current_official_api_index"]
    index = (ROOT / source["path"]).read_bytes()
    assert _sha256(index) == source["sha256"]
    decoded = index.decode("utf-8")
    for endpoint in source["required_endpoint_fragments"]:
        assert f"`{endpoint}`" in decoded
    assert (
        "`GET /sapi/v1/apiReferral/kickback/recentRecord`"
        " — Query Rebate Recent Record"
    ) in decoded
    assert (
        "`GET /sapi/v1/apiReferral/rebate/recentRecord`"
        " — Query Partner Rebate Recent Record"
    ) in decoded


def test_narrow_realized_client_kickback_overlay_is_fail_closed() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact, "result_sha256") == artifact["result_sha256"]
    decision = artifact["adjudication"]
    authority = artifact["authority"]

    assert decision["accepted_edge"] is True
    assert decision["market_direction_forecast_required"] is False
    assert decision["profitability_claim"] is False
    assert decision["deployment_ready"] is False
    assert decision["public_forward_profit_floor"] == "0"
    assert "Exact realized positive own-account" in decision["accepted_scope"]
    assert authority["credentials_used"] is False
    assert authority["signed_requests"] == 0
    assert authority["account_or_trade_history_accessed"] is False
    assert authority["accounts_linked_or_customized"] == 0
    assert authority["orders_or_trades"] == 0
    assert authority["state_changes"] == 0
    assert artifact["signed_read_only_reconciliation"][
        "state_changes_authorized"
    ] is False
    assert artifact["economic_contract"]["rate_assumption"] == "none"


def test_kickback_edge_updates_only_the_non_yield_population() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    frontier = json.loads(FRONTIER.read_text(encoding="utf-8"))

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _self_hash(frontier, "result_sha256") == frontier["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert family["priority_rank"] == 5
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    population = frontier["population"]
    assert population["registry_accepted_edge_count"] == registry[
        "accepted_edge_count"
    ]
    assert population["yield_and_capital_efficiency_edges_included"] == 9
    assert population[
        "organic_flow_fee_referral_creator_and_financing_cost_overlays_excluded"
    ] == (
        population["registry_accepted_edge_count"]
        - population["yield_and_capital_efficiency_edges_included"]
    )


def test_partner_rebate_is_a_distinct_fail_closed_realized_overlay() -> None:
    partner = json.loads(PARTNER_ARTIFACT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _self_hash(partner, "result_sha256") == partner["result_sha256"]
    decision = partner["adjudication"]
    authority = partner["authority"]
    assert decision["accepted_edge"] is True
    assert decision["accepted_edge_count_after"] == registry["accepted_edge_count"]
    assert decision["market_direction_forecast_required"] is False
    assert decision["owned_income_proved"] is False
    assert decision["profitability_claim"] is False
    assert decision["deployment_ready"] is False
    assert decision["public_forward_profit_floor"] == "0"
    assert authority["credentials_used"] is False
    assert authority["signed_requests"] == 0
    assert authority["state_changes"] == 0
    assert partner["economic_contract"]["rate_assumption"] == "none"
    assert partner["signed_read_only_reconciliation"][
        "state_changes_authorized"
    ] is False

    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "organic_third_party_platform_fee_overlays"
    )
    assert family["priority_rank"] == 24
    assert {
        "path": PARTNER_ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": partner["result_sha256"],
    } in family["canonical_artifacts"]
    assert "Link-and-Trade partner rebate income" in registry["accepted_edge_scope"]
    assert "must_not_be_retried" in family[
        "excluded_unretained_exchange_link_discovery"
    ]
