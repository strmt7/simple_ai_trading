from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
DELTA_CONTRACT = (
    ACTION_VALUE
    / "binance-crypto-option-population-delta-retained-contract-v1-2026-08-31.json"
)
DELTA_RESULT = (
    ACTION_VALUE
    / "binance-crypto-option-population-delta-retained-result-v1-2026-08-31.json"
)
PRICE_CONTRACT = (
    ACTION_VALUE
    / "binance-crypto-option-population-price-prefilter-contract-v1-2026-08-31.json"
)
PRICE_RESULT = (
    ACTION_VALUE
    / "binance-crypto-option-population-price-prefilter-result-v1-2026-08-31.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
JOURNAL = (
    ROOT
    / "data/binance-crypto-option-population-price-prefilter-v1/request-journal.jsonl"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_zero_network_delta_contract_and_result_are_hash_bound() -> None:
    contract = _load(DELTA_CONTRACT)
    result = _load(DELTA_RESULT)

    assert contract["status"] == "frozen_before_zero_network_population_delta"
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert contract["authority"]["new_public_requests"] == 0
    assert result["result_sha256"] == (
        "001abaada3b352235cbc38228dec6b6176a26cdfb33e208f0c3467f858cf9446"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert (
        result["contract"]["sha256"]
        == hashlib.sha256(DELTA_CONTRACT.read_bytes()).hexdigest()
    )


def test_delta_proves_an_exact_new_population_without_an_edge_claim() -> None:
    result = _load(DELTA_RESULT)
    population = result["population"]

    assert population["baseline_count"] == 1410
    assert population["current_count"] == 1576
    assert population["new_symbol_count"] == len(population["new_symbols"]) == 508
    assert (
        population["removed_symbol_count"] == len(population["removed_symbols"]) == 342
    )
    assert set(population["new_symbols"]).isdisjoint(population["removed_symbols"])
    assert result["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "literal_rank_47_new_population_trigger_satisfied": True,
        "next_action": "freeze_one_separate_public_price_prefilter_for_only_the_new_symbols",
        "profitability_claim": False,
    }


def test_price_contract_freezes_two_public_gets_and_fixed_rejection_gate() -> None:
    contract = _load(PRICE_CONTRACT)

    assert contract["status"] == "frozen_before_two_public_price_requests"
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert contract["authority"]["maximum_public_unauthenticated_GET_requests"] == 2
    assert contract["fixed_fee_and_basis_stress_bps"] == "33.5"
    assert [request["method"] for request in contract["requests"]] == ["GET", "GET"]
    assert (
        contract["retained_sources"]["population_delta"]["canonical_result_sha256"]
        == _load(DELTA_RESULT)["result_sha256"]
    )


def test_exact_new_population_has_zero_positive_gross_floors() -> None:
    result = _load(PRICE_RESULT)
    journal = [
        json.loads(line) for line in JOURNAL.read_text(encoding="ascii").splitlines()
    ]

    assert result["result_sha256"] == (
        "93d2ed3c9b6041f9ffcc7f9579f184687113049051a421f9fc048d2d4e309eee"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert (
        result["contract"]["sha256"]
        == hashlib.sha256(PRICE_CONTRACT.read_bytes()).hexdigest()
    )
    assert result["authority"]["public_unauthenticated_GET_requests"] == 2
    assert result["authority"]["authenticated_requests"] == 0
    assert result["authority"]["credentials_used"] is False
    assert result["population"] == {
        "after_fixed_stress_positive_count": 0,
        "gross_positive_count": 0,
        "new_symbol_count": 508,
        "positive_entry_side_count": 413,
    }
    assert len(result["all_rows"]) == 508
    assert result["fixed_stress_survivors"] == []
    assert max(
        Decimal(row["gross_terminal_floor_per_unit_USDT"])
        for row in result["all_rows"]
        if row["positive_entry_sides"]
    ) == Decimal("-0.38")
    assert result["adjudication"]["option_depth_requests"] == 0
    assert len(journal) == 2
    assert [row["url"] for row in journal] == [
        "https://eapi.binance.com/eapi/v1/ticker",
        "https://fapi.binance.com/fapi/v1/ticker/bookTicker",
    ]
    for receipt in journal:
        raw = ROOT / receipt["raw_path"]
        assert receipt["status_code"] == 200
        assert receipt["response_bytes"] == raw.stat().st_size
        assert (
            receipt["response_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
        )


def test_rank_47_and_terminal_registry_record_the_consumed_delta() -> None:
    registry = _load(REGISTRY)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_long_crypto_option_opposite_USDT_perpetual_terminal_payoff_lower_bound"
    )
    assert hypothesis["priority_rank"] == 47
    assert hypothesis["canonical_artifacts"][-1] == {
        "path": PRICE_RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": _load(PRICE_RESULT)["result_sha256"],
    }
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_BTC_ETH_SOL_long_option_opposite_perpetual_exact_new_population_2026_08_31"
    )
    assert terminal["canonical_result_sha256"] == _load(PRICE_RESULT)["result_sha256"]
