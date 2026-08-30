from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "docs/model-research/action-value/binance-euri-redemption-parity-retained-adjudication-v1-2026-08-30.json"
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(body, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_euri_source_and_retained_population_are_hash_bound() -> None:
    result = _load(RESULT_PATH)
    binding = result["source_binding"]
    source_contract = _load(ROOT / binding["issuer_source_contract_path"])
    source_result = _load(ROOT / binding["issuer_source_result_path"])
    parent_result = _load(ROOT / binding["retained_parent_result_path"])

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
        observations.append(next(row for row in rows if row["symbol"] == "EUREURI"))

    assert len(observations) == result["current_retained_population"]["book_observation_count"] == 60
    assert {row["bidPrice"] for row in observations} == {"1.00010000"}
    assert {row["askPrice"] for row in observations} == {"1.00030000"}
    assert {row["bidQty"] for row in observations} == {"41430.30000000"}
    assert {row["askQty"] for row in observations} == {"41838.70000000"}


def test_euri_family_is_terminal_without_inflating_accepted_edges() -> None:
    result = _load(RESULT_PATH)
    registry = _load(REGISTRY_PATH)
    family = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "binance_EURI_issuer_redemption_parity_current_retained_population_2026_08_30"
    )

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert family["canonical_result_sha256"] == result["result_sha256"]
    assert "independently_observed_event_driven_discount_above_25_bips" in family["reason"]


def test_euri_current_retained_economics_fail_before_redemption_costs() -> None:
    result = _load(RESULT_PATH)
    economics = result["economics_at_1000_eur"]
    bid = Decimal(result["current_retained_population"]["bid_price_euri_per_eur"])
    starting_eur = Decimal("1000")
    gross_euri = starting_eur * bid
    gross_profit_eur = gross_euri - starting_eur
    vip0_net_eur = gross_euri * (Decimal("1") - Decimal(economics["vip0_taker_fee_fraction"]))
    stress_eur = starting_eur * Decimal(economics["operational_stress_bips"]) / Decimal("10000")

    assert gross_euri == Decimal(economics["gross_euri_from_selling_eur_at_bid"])
    assert gross_profit_eur == Decimal(economics["gross_profit_eur_if_every_euri_redeems_at_par_with_zero_other_cost"])
    assert gross_profit_eur - stress_eur == Decimal(economics["profit_after_zero_fee_and_operational_stress_eur"])
    assert vip0_net_eur - starting_eur == Decimal(economics["vip0_profit_before_operational_stress_eur"])
    assert vip0_net_eur - starting_eur - stress_eur == Decimal(economics["vip0_profit_with_operational_stress_eur"])
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["stable_edge"] is False
