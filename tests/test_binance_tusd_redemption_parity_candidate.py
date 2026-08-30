from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "docs/model-research/action-value/binance-tusd-redemption-parity-retained-candidate-v1-2026-08-30.json"
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(body, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_tusd_source_and_retained_population_are_hash_bound() -> None:
    result = _load(RESULT_PATH)
    assert isinstance(result, dict)
    binding = result["source_binding"]
    assert isinstance(binding, dict)
    source_contract = _load(ROOT / binding["issuer_source_contract_path"])
    source_result = _load(ROOT / binding["issuer_source_result_path"])
    parent_result = _load(ROOT / binding["retained_parent_result_path"])
    assert isinstance(source_contract, dict)
    assert isinstance(source_result, dict)
    assert isinstance(parent_result, dict)

    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _self_hash(source_contract, "contract_sha256") == binding["issuer_source_contract_sha256"]
    assert _self_hash(source_result, "result_sha256") == binding["issuer_source_result_sha256"]
    assert source_result["source_gate"]["passed"] is True
    assert source_result["capture"]["receipt"]["response_sha256"] == binding["issuer_raw_response_sha256"]
    assert parent_result["result_sha256"] == binding["retained_parent_result_sha256"]

    raw_dir = ROOT / binding["retained_book_directory"]
    observations = []
    for path in sorted(raw_dir.glob("bookTicker-*.json")):
        rows = _load(path)
        assert isinstance(rows, list)
        observations.append(next(row for row in rows if row["symbol"] == "TUSDUSDT"))

    population = result["current_retained_population"]
    assert len(observations) == population["book_observation_count"] == 60
    assert {(row["bidPrice"], row["askPrice"]) for row in observations} == {
        ("0.99810000", "0.99830000")
    }
    assert min(Decimal(row["askQty"]) for row in observations) == Decimal(
        population["ask_quantity_min_tusd"]
    )
    assert max(Decimal(row["askQty"]) for row in observations) == Decimal(
        population["ask_quantity_max_tusd"]
    )


def test_tusd_current_retained_economics_are_only_a_cross_unit_sensitivity() -> None:
    result = _load(RESULT_PATH)
    assert isinstance(result, dict)
    economics = result["economics_at_1000_usdt"]
    ask = Decimal(result["current_retained_population"]["ask_price_usdt_per_tusd"])
    starting_usdt = Decimal("1000")
    gross_tusd = starting_usdt / ask
    fee_tusd = gross_tusd * Decimal(economics["vip0_taker_fee_fraction"])
    net_tusd = gross_tusd - fee_tusd
    stress = starting_usdt * Decimal(economics["operational_stress_bips"]) / Decimal("10000")

    assert gross_tusd == Decimal(economics["gross_tusd_acquired_at_ask"])
    assert fee_tusd == Decimal(economics["vip0_taker_fee_tusd"])
    assert net_tusd == Decimal(economics["net_tusd_after_vip0_fee"])
    assert net_tusd >= Decimal(economics["issuer_stated_minimum_redemption_tusd"])
    assert net_tusd - starting_usdt == Decimal(
        economics["profit_after_vip0_before_operational_stress_usd_sensitivity"]
    )
    assert net_tusd - starting_usdt - stress == Decimal(
        economics["profit_after_vip0_and_operational_stress_usd_sensitivity"]
    )
    assert result["adjudication"]["after_all_cost_profit_floor_usd"] == "0"
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["stable_edge"] is False


def test_tusd_event_driven_candidate_and_exact_population_terminal_are_registered() -> None:
    result = _load(RESULT_PATH)
    registry = _load(REGISTRY_PATH)
    assert isinstance(result, dict)
    assert isinstance(registry, dict)
    hypothesis = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"] == "binance_stablecoin_issuer_redemption_parity_event_dislocation"
    )
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "binance_TUSDUSDT_17_bip_retained_issuer_redemption_standalone_profit_claim_2026_08_30"
    )

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert hypothesis["priority_rank"] == 45
    assert "event_driven_TUSD_ask_discount_above_25_bips" in hypothesis["retry_trigger"]
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
