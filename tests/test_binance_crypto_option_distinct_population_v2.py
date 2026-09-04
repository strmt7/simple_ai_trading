from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
POPULATION = (
    ACTION_VALUE / "binance-crypto-option-population-gate-result-v2-2026-09-04.json"
)
PRICE = (
    ACTION_VALUE
    / "binance-crypto-option-distinct-price-prefilter-result-v2-2026-09-04.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ACTION_VALUE / "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_population_gate_is_self_bound_and_deduplicated() -> None:
    result = _load(POPULATION)

    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    population = result["population"]
    assert population["baseline_eligible_symbol_count"] == 1576
    assert population["current_eligible_symbol_count"] == 1488
    assert population["new_symbol_count"] == 356
    assert population["removed_symbol_count"] == 444
    assert population["previous_508_overlap_count"] == 0
    assert population["late_delta_overlap_symbols"] == [
        "BTC-261225-94000-C",
        "BTC-261225-94000-P",
    ]
    assert population["distinct_unscreened_symbol_count"] == 354
    assert population["distinct_unscreened_underlying_counts"] == {
        "BTCUSDT": 174,
        "ETHUSDT": 90,
        "SOLUSDT": 90,
    }
    assert result["preflight_correction"]["network_requests_before_correction"] == 0
    assert result["preflight_correction"]["request_changed"] is False


def test_price_prefilter_rejects_the_only_gross_positive_row() -> None:
    result = _load(PRICE)

    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["capture"]["capture_skew_ms"] == 245
    assert result["population"] == {
        "after_fixed_stress_positive_count": 0,
        "distinct_unscreened_symbol_count": 354,
        "gross_positive_count": 1,
        "positive_entry_side_count": 226,
        "ticker_present_count": 354,
    }
    gross_positive = [
        row
        for row in result["all_rows"]
        if row["positive_entry_sides"]
        and Decimal(row["gross_terminal_floor_per_underlying_unit_USDT"]) > 0
    ]
    assert len(gross_positive) == 1
    row = gross_positive[0]
    assert row["symbol"] == "BTC-260905-80500-P"
    assert row["gross_terminal_floor_per_underlying_unit_USDT"] == "11.10"
    assert row["after_fixed_stress_per_underlying_unit_USDT"] == "-255.455815"
    assert row["passes_fixed_rejection_gate"] is False
    assert result["fixed_stress_survivors"] == []


def test_canonical_ledgers_terminalize_only_this_population() -> None:
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert registry["result_sha256"] == (
        "51dba2cf9c5f61efa650cab4db66edc37189cd9aeb7ddfd97dbfe9fdd0e16698"
    )
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert len(registry["prioritized_hypotheses"]) == 65
    assert len(registry["terminal_do_not_repeat"]) == 183
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_long_crypto_option_opposite_USDT_perpetual_terminal_payoff_lower_bound"
    )
    assert hypothesis["priority_rank"] == 47
    assert hypothesis["canonical_artifacts"][-1] == {
        "path": PRICE.relative_to(ROOT).as_posix(),
        "result_sha256": _load(PRICE)["result_sha256"],
    }
    assert audit["source_binding"]["registry_result_sha256"] == registry[
        "result_sha256"
    ]
    assert audit["result_sha256"] == (
        "e575d0084546053bc1ab2c07e9d23553796d4ef4dd8fc29bc2edcef314575801"
    )
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
