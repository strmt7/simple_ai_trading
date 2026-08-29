from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / (
    "binance-xau-paxg-perpetual-funding-spread-contract-v1-2026-08-29.json"
)
RESULT = ACTION_VALUE / (
    "binance-xau-paxg-perpetual-funding-spread-result-v1-2026-08-29.json"
)
ADJUDICATION = ACTION_VALUE / (
    "binance-xau-paxg-perpetual-funding-spread-failure-adjudication-v1-2026-08-29.json"
)
SCREEN = ROOT / "tools/screen_binance_xau_paxg_perpetual_funding_spread.py"
ADJUDICATOR = ROOT / "tools/adjudicate_binance_xau_paxg_perpetual_funding_spread.py"
JOURNAL = ROOT / (
    "data/binance-xau-paxg-perpetual-funding-spread-v1/request-journal.jsonl"
)
RAW = {
    "XAUUSDT": ROOT
    / "data/binance-xau-paxg-perpetual-funding-spread-v1/raw/funding-xauusdt.json",
    "PAXGUSDT": ROOT
    / "data/binance-xau-paxg-perpetual-funding-spread-v1/raw/funding-paxgusdt.json",
}
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "39367c3544711a6c206e8d9a3b98f1832ed2c912d8f113ad35872a1fb11e6f36"
RESULT_HASH = "4cf430a7c5b6ce6ab57fd71979d705732e737cbcb116658705454b80daa025a9"
ADJUDICATION_HASH = "46bf134d1be8b645d7f6272d651be8d3c0b6a8e5b2e7d2b4540f3609d6997a96"
REGISTRY_HASH = "2baf1b76070e0ef9081f9eb5fba41f3977b5fd1aa74759ed85034947e9ad1c5a"


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


def test_frozen_contract_is_public_historical_and_action_free() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert contract["historical_contract"]["maximum_requests"] == 2
    assert contract["historical_contract"]["retry_permitted"] is False
    assert contract["historical_contract"]["immutable_global_cutoff_exclusive_utc"] == (
        "2026-08-14T00:00:00Z"
    )
    assert (
        contract["authority"]["credentials_accounts_orders_or_positions_permitted"]
        is False
    )
    assert contract["causal_analysis"]["capital_legs"] == "2"
    assert contract["causal_analysis"]["selection"] == (
        "choose the direction with the higher training net only"
    )


def test_original_failure_and_retained_sources_are_immutable() -> None:
    result = _load(RESULT)
    receipts = [
        json.loads(line) for line in JOURNAL.read_text(encoding="ascii").splitlines()
    ]

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["contract"]["sha256"] == CONTRACT_HASH
    assert result["capture"]["XAUUSDT_returned_row_count"] == 500
    assert result["capture"]["PAXGUSDT_returned_row_count"] == 500
    assert result["capture"]["exact_common_regular_row_count"] == 83
    assert result["capture"]["complete_common_history"] is False
    assert result["adjudication"]["accepted_edge"] is False
    assert result["authority"]["public_unauthenticated_GET_requests"] == 2
    assert result["authority"]["authenticated_requests"] == 0
    assert result["authority"]["positions_or_orders"] == 0
    assert len(receipts) == 2
    assert all(row["method"] == "GET" and row["status_code"] == 200 for row in receipts)
    assert receipts[1]["requested_at_ms"] - receipts[0]["requested_at_ms"] >= 650
    for receipt in receipts:
        symbol = "XAUUSDT" if "xauusdt" in receipt["name"] else "PAXGUSDT"
        assert (
            receipt["response_sha256"]
            == hashlib.sha256(RAW[symbol].read_bytes()).hexdigest()
        )
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(SCREEN.read_bytes()).hexdigest()
    )


def test_retained_complete_ordinal_tail_rejects_both_directions_everywhere() -> None:
    adjudication = _load(ADJUDICATION)

    assert adjudication["result_sha256"] == ADJUDICATION_HASH
    assert _canonical_hash(adjudication, "result_sha256") == ADJUDICATION_HASH
    assert (
        adjudication["implementation"]["sha256"]
        == hashlib.sha256(ADJUDICATOR.read_bytes()).hexdigest()
    )
    diagnosis = adjudication["failure_diagnosis"]
    assert diagnosis["exact_millisecond_intersection_count"] == 83
    assert diagnosis["ordinal_pair_count"] == 500
    assert diagnosis["maximum_ordinal_timestamp_skew_ms"] == 13
    sensitivity = adjudication["retained_evidence_sensitivity"]
    assert sensitivity["all_directions_fail_every_role"] is True
    assert sensitivity["selected_direction_from_training_only"] == (
        "long_PAXG_short_XAU"
    )
    for role in ("training", "validation", "test"):
        assert len(sensitivity["directions"][role]) == 2
        assert all(
            Decimal(row["net_after_frozen_hurdles_bips"]) < 0 and row["passes"] is False
            for row in sensitivity["directions"][role].values()
        )
    assert adjudication["adjudication"] == {
        "accepted_edge": False,
        "candidate_for_prospective_study": False,
        "deployment_ready": False,
        "profitability_claim": False,
        "rerun_or_pagination_justified": False,
        "retry_trigger": "material_funding_index_fee_or_product_architecture_change",
        "status": "rejected_under_retained_complete_500_slot_tail",
        "trading_authority": False,
    }


def test_registry_terminalizes_the_failed_gold_spread_without_acceptance() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 43)
    )
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["priority_rank"] == 41
    )
    assert row["priority_rank"] == 41
    assert row["mechanism"] == (
        "binance_XAUUSDT_PAXGUSDT_same_venue_perpetual_funding_basis_spread"
    )
    assert [item["result_sha256"] for item in row["canonical_artifacts"]] == [
        CONTRACT_HASH,
        RESULT_HASH,
        ADJUDICATION_HASH,
    ]
    assert "do_not_rerun_paginate_or_refit" in row["next_action"]
