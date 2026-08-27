from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "polymarket-live-nba-moneyline-spread-combinatorial-parity-reopen-v1-"
    "2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "70cfc7b2ae1cb256e7a8c08c9af33fa8524d2308a8c18400d5a2b7d93c966fe3"
REGISTRY_HASH = "51005178e370b03e2974b20e780b8ed5b25b7847b59b8c284fb7bca9ea12c70f"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_live_nba_implication_is_hash_bound_distinct_candidate_only() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "status": (
            "materially_reopened_distinct_live_NBA_moneyline_spread_implication_"
            "candidate_current_fee_recurrence_and_exact_settlement_contract_required"
        ),
        "trading_authority": False,
    }
    assert artifact["authority"]["market_catalog_or_order_book_requests_sent"] == 0


def test_current_fee_regime_and_push_state_block_historical_profit_claim() -> None:
    artifact = _load(ARTIFACT)
    fees = artifact["current_fee_break_even"]
    examples = fees["illustrative_not_evidence"]

    assert Decimal(fees["current_category_reference_only_sports_fee_rate"]) == Decimal(
        "0.05"
    )
    assert Decimal(examples[0]["net_before_external_costs_per_share"]) == Decimal(
        "-0.014995"
    )
    assert Decimal(examples[1]["net_before_external_costs_per_share"]) == Decimal(
        "0.008525"
    )
    assert "zero NBA trading fees" in artifact["historical_primary_evidence"]["warning"]
    assert "push" in artifact["payoff_identity"]["required_push_tie_overtime_gate"]
    assert artifact["payoff_identity"]["middle_value_in_acceptance"] == "0"


def test_registry_reopens_only_the_distinct_future_nba_recurrence_family() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 17
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 41)
    )
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_live_NBA_moneyline_spread_monotone_payoff_implication"
    )
    assert row["priority_rank"] == 30
    assert row["mechanism"] == (
        "polymarket_live_NBA_moneyline_spread_monotone_payoff_implication"
    )
    assert row["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-live-nba-moneyline-spread-combinatorial-parity-reopen-"
                "v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        },
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-sports-taker-delay-maker-protection-gate-v1-"
                "2026-08-27.json"
            ),
            "result_sha256": (
                "4847ec7828e598950da9a455170b66a529d9a5d671bfb4c37a57a36f608b9627"
            ),
        },
    ]
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"] == "polymarket_threshold_and_deadline_implication_parity"
    )
    assert terminal["canonical_result_sha256"] == (
        "c77c5c6e2e525898f334bd81c54d1b60673226b7488b2833f2f15e17e4de1f78"
    )
