from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.adjudicate_binance_cross_sectional_funding_dispersion import (  # noqa: E402
    TEN_THOUSAND,
    _position,
    _role_result,
)


BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "binance-btc-eth-sol-cross-sectional-funding-dispersion-contract-v1-2026-08-30.json"
)
RESULT = BASE / (
    "binance-btc-eth-sol-cross-sectional-funding-dispersion-result-v1-2026-08-30.json"
)
SENSITIVITY = BASE / (
    "binance-btc-eth-sol-cross-sectional-funding-dispersion-capital-sensitivity-"
    "v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _intervals(contract: dict[str, object]) -> list[dict[str, object]]:
    series: dict[str, list[dict[str, object]]] = {}
    for source in contract["sources"]:
        path = ROOT / source["path"]
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["file_sha256"]
        series[source["asset"]] = sorted(
            json.loads(payload), key=lambda row: row["fundingTime"]
        )

    assets = contract["population"]["assets"]
    result: list[dict[str, object]] = []
    for index in range(1, contract["population"]["source_row_count"]):
        result.append(
            {
                "lagged_rates": {
                    asset: Decimal(series[asset][index - 1]["fundingRate"])
                    for asset in assets
                },
                "realized_rates": {
                    asset: Decimal(series[asset][index]["fundingRate"])
                    for asset in assets
                },
            }
        )
    return result


def test_frozen_sources_contract_result_and_implementation_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    result = json.loads(RESULT.read_text(encoding="ascii"))

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    implementation = contract["implementation"]
    assert (
        hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
        == (implementation["sha256"])
    )
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert result["implementation"]["sha256"] == implementation["sha256"]
    assert len(_intervals(contract)) == 209


def test_causal_roles_and_perfect_foresight_bound_reconstruct() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    result = json.loads(RESULT.read_text(encoding="ascii"))
    intervals = _intervals(contract)
    per_interval_capital = (
        Decimal(2)
        * Decimal(contract["economics"]["annual_capital_hurdle_percent_per_leg"])
        / Decimal(100)
        * Decimal(contract["population"]["interval_hours"])
        / Decimal(24)
        / Decimal(365)
        * TEN_THOUSAND
    )

    for role_index, (role, bounds) in enumerate(contract["roles"].items()):
        subset = intervals[bounds["start"] : bounds["stop"]]
        actual = _role_result(
            subset, contract, seed=contract["bootstrap"]["seed"] + role_index
        )
        assert actual == result["causal_lagged_dispersion_strategy"][role]

        oracle_gross = Decimal(0)
        for interval in subset:
            long_asset, short_asset, _ = _position(interval["realized_rates"])
            oracle_gross += (
                interval["realized_rates"][short_asset]
                - interval["realized_rates"][long_asset]
            ) * TEN_THOUSAND
        capital = per_interval_capital * Decimal(len(subset))
        oracle = result["perfect_foresight_dominance_bound"][role]
        assert oracle_gross == Decimal(
            oracle["perfect_foresight_zero_execution_gross_funding_bips"]
        )
        assert oracle_gross - capital == Decimal(
            oracle["perfect_foresight_zero_execution_net_after_two_leg_capital_bips"]
        )
        assert oracle["clears_two_leg_capital_hurdle"] is False


def test_terminal_rejection_blocks_price_requests_without_acceptance() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    sensitivity = json.loads(SENSITIVITY.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _self_hash(sensitivity, "result_sha256") == sensitivity["result_sha256"]
    assert sensitivity["source_binding"]["parent_result_canonical_sha256"] == (
        result["result_sha256"]
    )
    assert (
        sensitivity["gates"][
            "fixed_orientation_rescue_clears_super_optimistic_5x_bound"
        ]
        is False
    )
    assert Decimal(
        sensitivity["calculations_bips"][
            "super_optimistic_combined_holdout_net_at_5x"
        ]
    ) < 0
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["stable_profitability_proved"] is False
    assert result["gates"]["price_capture_justified"] is False
    assert result["authority"]["books_or_price_requests"] == 0
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 0

    family = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"] == "binance_BTC_ETH_SOL_same_venue_cross_sectional_"
        "perpetual_funding_dispersion_2026_08_30"
    )
    assert family["canonical_result_sha256"] == sensitivity["result_sha256"]
    assert "perfect_foresight" in family["reason"]
