from __future__ import annotations

from itertools import product
import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_crypto_interval_composition import _json_list
from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import _canonical_hash


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT_V1 = ACTION / (
    "polymarket-crypto-interval-composition-contract-v1-2026-08-30.json"
)
FAILURE_V1 = ACTION / (
    "polymarket-crypto-interval-composition-v1-pre-economic-failure-2026-08-30.json"
)
CONTRACT_V2 = ACTION / (
    "polymarket-crypto-interval-composition-contract-v2-2026-08-30.json"
)
RESULT_V2 = ACTION / (
    "polymarket-crypto-interval-composition-result-v2-2026-08-30.json"
)
RESULT_V1 = ACTION / (
    "polymarket-crypto-interval-composition-result-v1-2026-08-30.json"
)
TRADE_CONTRACT = ACTION / (
    "polymarket-crypto-interval-composition-trades-contract-v1-2026-08-30.json"
)
TRADE_FAILURE = ACTION / (
    "polymarket-crypto-interval-composition-trades-failure-v1-2026-08-30.json"
)
SETTLEMENT_CONTRACT = ACTION / (
    "polymarket-crypto-interval-composition-settlement-contract-v1-2026-08-30.json"
)
SETTLEMENT_RESULT = ACTION / (
    "polymarket-crypto-interval-composition-settlement-result-v1-2026-08-30.json"
)
RUNNER_LINEAGE = ACTION / (
    "polymarket-crypto-interval-composition-runner-lineage-v1-2026-08-30.json"
)
TRADE_RAW = (
    ROOT / "data/polymarket-crypto-interval-composition-trades-v1/raw/trades.json"
)
TRADE_JOURNAL = (
    ROOT / "data/polymarket-crypto-interval-composition-trades-v1/request-journal.jsonl"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def test_transitive_packages_pay_at_least_one_in_every_order_state() -> None:
    for x0, x1, x2, x3 in product((-1, 0, 1), repeat=4):
        short_up = (x1 >= x0, x2 >= x1, x3 >= x2)
        long_up = x3 >= x0
        up_chain_payout = sum(not value for value in short_up) + long_up
        down_chain_payout = sum(short_up) + (not long_up)
        assert up_chain_payout >= 1
        assert down_chain_payout >= 1


def test_gamma_json_list_decoder_handles_retained_representation() -> None:
    assert _json_list('["Up","Down"]', name="outcomes") == ["Up", "Down"]
    assert _json_list(["Up", "Down"], name="outcomes") == ["Up", "Down"]
    with pytest.raises(RuntimeError, match="must be a JSON list"):
        _json_list('{"Up":1}', name="outcomes")


def test_corrected_complete_screen_rejects_all_six_packages() -> None:
    contract_v1 = _load(CONTRACT_V1)
    failure_v1 = _load(FAILURE_V1)
    contract_v2 = _load(CONTRACT_V2)
    result = _load(RESULT_V2)

    assert (
        _canonical_hash(contract_v1, "contract_sha256")
        == contract_v1["contract_sha256"]
    )
    assert _canonical_hash(failure_v1, "result_sha256") == failure_v1["result_sha256"]
    assert (
        _canonical_hash(contract_v2, "contract_sha256")
        == contract_v2["contract_sha256"]
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert not RESULT_V1.exists()
    assert failure_v1["failure"]["economic_price_values_accessed"] is False
    assert failure_v1["authority"]["network_requests"] == 0
    assert result["screen"]["market_count"] == 12
    assert result["screen"]["package_count"] == 6
    assert result["screen"]["strict_displayed_candidate_count"] == 0
    assert [
        row["displayed_price_sum_pUSD"]
        for row in result["screen"]["packages_ranked_by_displayed_sum"]
    ] == ["1.990", "1.990", "1.990", "2.010", "2.010", "2.010"]
    assert result["screen"]["best_package"]["asset"] == "BTC"
    assert result["screen"]["best_package"]["direction"] == "up_chain"
    assert result["authority"]["network_requests"] == 0
    assert result["adjudication"]["accepted_edge"] is False


def test_registry_routes_interval_composition_to_rank_31_and_terminal() -> None:
    registry = _load(REGISTRY)
    result = _load(RESULT_V2)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_31 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    assert any(
        artifact["result_sha256"] == result["result_sha256"]
        for artifact in rank_31["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_BTC_ETH_SOL_August_31_11_45_to_12_00_TWAP_interval_composition_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert "50_packages_with_zero_floor_violations" in terminal["reason"]


def test_consumed_trade_batch_is_hash_bound_terminal_and_not_retried() -> None:
    contract = _load(TRADE_CONTRACT)
    failure = _load(TRADE_FAILURE)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(failure, "result_sha256") == failure["result_sha256"]
    assert failure["contract"]["sha256"] == contract["contract_sha256"]
    assert failure["capture"]["request_count"] == 1
    assert failure["capture"]["http_status"] == 408
    assert failure["capture"]["trade_rows_exposed"] == 0
    assert failure["capture"]["result_output_created"] is False
    assert failure["adjudication"]["retry_allowed"] is False
    assert failure["authority"]["network_requests"] == 1
    assert failure["authority"]["credentials_used"] is False
    assert failure["authority"]["orders_or_transactions"] == 0
    assert (
        hashlib.sha256(TRADE_RAW.read_bytes()).hexdigest()
        == failure["capture"]["raw_sha256"]
    )
    assert (
        hashlib.sha256(TRADE_JOURNAL.read_bytes()).hexdigest()
        == failure["capture"]["journal_sha256"]
    )
    assert json.loads(TRADE_RAW.read_text(encoding="ascii")) == {
        "error": "Request timed out. Please try again."
    }


def test_retained_settlements_support_every_package_payoff_floor() -> None:
    contract = _load(SETTLEMENT_CONTRACT)
    result = _load(SETTLEMENT_RESULT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert result["population"] == {
        "assets": ["BTC", "ETH", "SOL"],
        "complete_set_count": 25,
        "complete_set_keys_sha256": (
            "17e8486a7c626196db8df80a325c0b3a63d6e98871b8e7423ba26d045275dcb9"
        ),
        "market_count": 100,
        "package_evaluation_count": 50,
    }
    assert len(result["audit"]["rows"]) == 25
    assert result["audit"]["minimum_realized_package_payout_pUSD"] == "1"
    assert result["audit"]["maximum_realized_package_payout_pUSD"] == "3"
    assert result["audit"]["floor_violation_count"] == 0
    assert result["audit"]["all_package_payoff_floors_satisfied"] is True
    for row in result["audit"]["rows"]:
        short_up = [winner == "Up" for winner in row["short_interval_winners"]]
        long_up = row["long_interval_winner"] == "Up"
        assert int(row["up_chain_payout_pUSD_per_package"]) == sum(
            not value for value in short_up
        ) + int(long_up)
        assert int(row["down_chain_payout_pUSD_per_package"]) == sum(short_up) + int(
            not long_up
        )
    assert result["authority"]["network_requests"] == 0
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["profitability_claim"] is False


def test_consumed_runner_bytes_remain_exactly_reconstructable() -> None:
    lineage = _load(RUNNER_LINEAGE)

    assert _canonical_hash(lineage, "result_sha256") == lineage["result_sha256"]
    assert (
        lineage["change_classification"]
        == "ruff_format_only_no_logic_or_outcome_change"
    )
    for row in lineage["consumed_implementations"]:
        sidecar = ROOT / row["immutable_sidecar_path"]
        current = ROOT / row["current_formatted_path"]
        assert (
            hashlib.sha256(sidecar.read_bytes()).hexdigest() == row["expected_sha256"]
        )
        assert row["immutable_sidecar_sha256"] == row["expected_sha256"]
        assert (
            hashlib.sha256(current.read_bytes()).hexdigest()
            == row["current_formatted_sha256"]
        )
        assert row["current_formatted_sha256"] != row["expected_sha256"]


def test_registry_preserves_trade_failure_and_settlement_routing() -> None:
    registry = _load(REGISTRY)
    failure = _load(TRADE_FAILURE)
    settlement = _load(SETTLEMENT_RESULT)
    rank_31 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    artifact_hashes = {
        artifact["result_sha256"] for artifact in rank_31["canonical_artifacts"]
    }

    assert failure["result_sha256"] in artifact_hashes
    assert settlement["result_sha256"] in artifact_hashes
    assert (
        "future_distinct_nonconsumed_exactly_aligned_interval_partition"
        in rank_31["next_action"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_BTC_ETH_SOL_August_26_48_condition_interval_composition_historical_taker_trade_batch_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == failure["result_sha256"]
    assert "do_not_retry_split_narrow_paginate_reorder_or_alias" in terminal["reason"]
