from __future__ import annotations

from itertools import product
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

    assert _canonical_hash(contract_v1, "contract_sha256") == contract_v1["contract_sha256"]
    assert _canonical_hash(failure_v1, "result_sha256") == failure_v1["result_sha256"]
    assert _canonical_hash(contract_v2, "contract_sha256") == contract_v2["contract_sha256"]
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
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 31
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
    assert "best_BTC_up_chain_cost_1_990_pUSD" in terminal["reason"]
