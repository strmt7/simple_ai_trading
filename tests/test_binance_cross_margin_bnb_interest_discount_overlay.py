from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-cross-margin-bnb-interest-discount-overlay-v1-2026-08-30.json"
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


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return hashlib.sha256(_canonical(body)).hexdigest()


def test_artifact_reconstructs_and_binds_current_primary_contracts() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert _self_hash(artifact) == artifact["result_sha256"]

    sources = {row["role"]: row for row in artifact["sources"]}
    rate = sources["current_official_discount_rate_scope_and_BNB_balance_requirement"]
    status = sources["current_official_read_only_enabled_status_contract"]
    history = sources["current_official_exact_realized_converted_interest_evidence_contract"]

    assert rate["facts"]["cross_margin_interest_discount_fraction"] == "0.05"
    assert rate["raw_response_sha256"] == (
        "94e4ad2879f859a18db83fe13e9f790b1fc3b72c3c3cccae4f0d965bc9b7cdac"
    )
    assert status["facts"]["endpoint"] == "GET /sapi/v1/bnbBurn"
    assert status["facts"]["response_field"] == "interestBNBBurn"
    assert history["facts"]["endpoint"] == "GET /sapi/v1/margin/interestHistory"
    assert history["facts"]["converted_cross_margin_types"] == [
        "PERIODIC_CONVERTED",
        "ON_BORROW_CONVERTED",
    ]


def test_discount_is_scoped_to_existing_cross_margin_cost_only() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    decision = artifact["adjudication"]
    economics = artifact["economic_contract"]

    assert decision["accepted_edge"] is True
    assert decision["market_direction_forecast_required"] is False
    assert decision["fresh_borrow_or_trade_profit_proved"] is False
    assert decision["public_forward_profit_floor_quote_units"] == "0"
    assert economics["discount_fraction"] == "0.05"
    assert economics["isolated_margin_included"] is False
    assert economics["portfolio_margin_included"] is False
    assert economics["underlying_borrow_profitability_accepted"] is False
    assert artifact["authority"]["official_venue_API_requests"] == 0
    assert artifact["authority"]["orders_borrows_repays_or_mutations"] == 0


def test_registry_accepts_only_the_scoped_overlay_and_binds_artifact() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry) == registry["result_sha256"]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert family["mechanism"] == "binance_spot_fee_minimization_overlays"
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "exact 5 percent Binance Cross Margin interest reduction" in registry[
        "accepted_edge_scope"
    ]
