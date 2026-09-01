from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
EDGE = ACTION / "predictfun-maker-rebate-and-cross-venue-adjudication-v1-2026-09-01.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ACTION / "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object], field: str = "result_sha256") -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_predictfun_sources_and_math_are_bound() -> None:
    edge = _load(EDGE)
    assert _self_hash(edge) == edge["result_sha256"]

    for source in edge["sources"]:
        result = _load(ROOT / source["path"])
        raw = ROOT / result["capture"]["receipt"]["raw_path"]
        assert _self_hash(result) == result["result_sha256"]
        assert result["result_sha256"] == source["result_sha256"]
        assert _sha256(raw) == source["raw_sha256"]
        assert result["source_gate"]["passed"] is True

    math = edge["fee_and_rebate_math"]
    assert math["published_base_fee_rate_without_discount"] == 0.02
    assert edge["current_program_terms"]["maker_rebate_share_of_taker_fee"] == 0.25
    assert math["maximum_nominal_rebate_per_share_usdt_equivalent_without_discount"] == 0.0025
    assert math["maximum_nominal_rebate_per_share_usdt_equivalent_with_discount"] == 0.00225


def test_predictfun_edge_is_scoped_and_false_parities_are_terminal() -> None:
    edge = _load(EDGE)
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert _self_hash(registry) == registry["result_sha256"]
    assert _self_hash(audit) == audit["result_sha256"]
    assert edge["adjudication"]["accepted_edge"] is True
    assert edge["adjudication"]["underlying_market_making_strategy_accepted"] is False
    assert edge["adjudication"]["cross_venue_arbitrage_accepted"] is False
    assert edge["adjudication"]["public_forward_profit_floor_usdt"] == 0
    assert registry["accepted_edge_count"] == 31
    assert audit["source_binding"]["accepted_edge_count"] == 31

    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "predictfun_eligible_crypto_maker_rebate_overlay"
    )
    assert hypothesis["market_direction_forecast_required"] is False
    assert hypothesis["priority_rank"] == 48
    assert hypothesis["canonical_artifacts"][0]["result_sha256"] == edge["result_sha256"]

    tier_c = next(
        group
        for group in audit["classification"]
        if group["evidence_tier"] == "C_preexisting_internal_activity_saving_or_rebate"
    )
    assert 31 in tier_c["edge_ordinals"]

    terminal = {row["family"] for row in registry["terminal_do_not_repeat"]}
    assert "predictfun_Predict_Points_liquidity_provision_monetary_value_2026_09_01" in terminal
    assert "predictfun_Polymarket_BTC_five_minute_cross_venue_payoff_identity_2026_09_01" in terminal
    assert "predictfun_Polymarket_OpenAI_acquired_before_2027_cross_venue_ask_package_2026_09_01" in terminal

    assert edge["authority"]["credentials_used"] is False
    assert edge["authority"]["orders_or_transactions"] == 0
