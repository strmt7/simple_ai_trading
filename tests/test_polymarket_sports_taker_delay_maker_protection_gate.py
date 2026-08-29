from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "polymarket-sports-taker-delay-maker-protection-gate-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
SOURCE = ROOT / "src/simple_ai_trading/polymarket.py"
EXPECTED_HASH = "4847ec7828e598950da9a455170b66a529d9a5d671bfb4c37a57a36f608b9627"
REGISTRY_HASH = "7b03ee420b7874180732f28fc1c59903adbd82e147be268ea54e894f460bbda1"


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


def test_sports_delay_gate_is_hash_bound_and_non_promoting() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["authority"]["authenticated_requests"] == 0
    assert artifact["authority"]["venue_market_data_or_API_requests"] == 0
    assert artifact["authority"]["orders_cancellations_or_quotes_submitted"] == 0
    assert not artifact["adjudication"]["accepted_edge"]
    assert artifact["adjudication"]["public_after_cost_profit_floor_pusd"] == "0"


def test_pending_delay_and_source_conflicts_fail_closed() -> None:
    artifact = _load(ARTIFACT)
    evidence = artifact["current_primary_rule_evidence"]
    delay = evidence["marketable_order_delay"]

    assert delay["general_sports_help_center_seconds"] == 3
    assert delay["NBA_and_MLB_test_seconds"] == 1
    assert delay["pending_cancellation_rule"] == (
        "during_the_delay_the_marketable_order_is_pending_and_cannot_be_cancelled"
    )
    assert not delay["exact_current_per_market_duration_publicly_exposed"]
    assert evidence["compact_market_info_limit"]["sports_numeric_delay_field_documented"] is False
    assert evidence["sports_maker_rebate_conflict"] == {
        "help_trading_fees_page_rate": "0.15",
        "help_maker_rebates_page_rate": "0.20",
        "resolution": (
            "unresolved_current_primary_source_conflict_credit_zero_until_an_"
            "effective_source_or_owned_post_change_payout_resolves_it"
        ),
    }


def test_registry_and_crypto_constant_keep_sports_separate() -> None:
    registry = _load(REGISTRY)
    artifact = _load(ARTIFACT)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_live_NBA_moneyline_spread_monotone_payoff_implication"
    )
    assert row["canonical_artifacts"][-1] == {
        "path": (
            "docs/model-research/action-value/"
            "polymarket-sports-taker-delay-maker-protection-gate-v1-2026-08-27.json"
        ),
        "result_sha256": EXPECTED_HASH,
    }
    assert artifact["execution_consequences"]["net_effect"].endswith(
        "not_250_milliseconds"
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert "Sports/game delays use a separate configured" in source
    assert "must never inherit this value" in source
