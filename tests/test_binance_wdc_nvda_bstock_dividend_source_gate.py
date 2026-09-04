from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / (
    "binance-wdc-nvda-bstock-dividend-source-contract-v1-2026-09-04.json"
)
SOURCE_RESULT = ACTION_VALUE / (
    "binance-wdc-nvda-bstock-dividend-source-result-v1-2026-09-04.json"
)
ADJUDICATION = ACTION_VALUE / (
    "binance-wdc-nvda-bstock-dividend-net-floor-adjudication-v1-2026-09-04.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ACTION_VALUE / (
    "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)

CONTRACT_HASH = "36019fbf294ae3f3182ed75c6a4e35e1f305fd1ef7d44e59d24609ca663e3f6f"
SOURCE_RESULT_HASH = (
    "fd294181f625d62967254bc97de6f55930b6af9194b97cc9280f493c82702e21"
)
ADJUDICATION_HASH = (
    "82bd2e0b0461b930218da3b7e01756cc2a5572d823bfc49586421fa3a7d5ce98"
)


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


def test_source_contract_and_result_are_hash_bound_and_one_use() -> None:
    contract = _load(CONTRACT)
    result = _load(SOURCE_RESULT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert contract["request"]["count"] == 1
    assert contract["trigger"]["full_retry_trigger_satisfied_before_freeze"] is False
    assert contract["economic_decision"]["market_data_request_if_gate_fails"] is False

    assert result["result_sha256"] == SOURCE_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == SOURCE_RESULT_HASH
    assert result["source_gate"]["passed"] is True
    assert result["capture"]["receipt"]["response_bytes"] == 69675
    assert result["authority"]["credentials_used"] is False
    assert result["authority"]["protected_capture_touched"] is False


def test_both_siblings_fail_before_market_data() -> None:
    adjudication = _load(ADJUDICATION)

    assert adjudication["result_sha256"] == ADJUDICATION_HASH
    assert _canonical_hash(adjudication, "result_sha256") == ADJUDICATION_HASH
    terms = adjudication["shared_exact_terms"]
    assert terms["exact_positive_gross_cash_amount_per_bstock_unit_published"] is False
    assert terms["finite_withholding_upper_bound_or_complete_formula_published"] is False
    assert (
        terms[
            "finite_fee_cost_and_other_deduction_upper_bound_or_complete_formula_published"
        ]
        is False
    )
    assert terms["deterministic_positive_final_units_or_multiplier_increment_published"] is False
    assert terms["currency_amount_candidates_in_article_body"] == []

    assert [row["bstock_symbol"] for row in adjudication["episodes"]] == [
        "WDCB",
        "NVDAB",
    ]
    assert all(
        row["public_conservative_net_distribution_floor_usd"] == "0"
        for row in adjudication["episodes"]
    )
    gate = adjudication["gate_outcome"]
    assert gate["full_rank_34_retry_trigger_satisfied"] is False
    assert gate["books_or_funding_justified"] is False
    assert adjudication["authority"]["new_public_requests"] == 1
    assert adjudication["authority"]["authenticated_requests"] == 0


def test_registry_and_durability_audit_route_the_terminal_episode() -> None:
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_bstock_dividend_perpetual_funding_timing_gap"
    )
    assert hypothesis["priority_rank"] == 34
    assert any(
        artifact["result_sha256"] == ADJUDICATION_HASH
        for artifact in hypothesis["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_WDCB_NVDAB_cash_dividend_shared_net_floor_source_gate_2026_09_04"
    )
    assert terminal["canonical_result_sha256"] == ADJUDICATION_HASH

    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    assert audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    assert "WDCB and NVDAB" in audit["routing"][
        "binance_wdc_nvda_bstock_dividend_source_floor_terminal_trigger"
    ]
