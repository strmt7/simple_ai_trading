"""Adjudicate one retained-data BTC/ETH/SOL funding-dispersion preflight."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import random
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)


SCHEMA = "binance-cross-sectional-funding-dispersion-preflight-v1"
ZERO = Decimal(0)
TEN_THOUSAND = Decimal(10_000)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{path.name} must contain a nonempty array")
    if any(not isinstance(row, dict) for row in value):
        raise RuntimeError(f"{path.name} contains a non-object row")
    return value


def _validate_contract(contract: dict[str, Any], path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    if contract.get("network_requests") != 0:
        raise RuntimeError("network boundary changed")
    output = _root_path(contract["output_path"])
    if output.exists():
        raise RuntimeError("one-use output already exists")
    implementation = contract["implementation"]
    implementation_path = _root_path(implementation["path"])
    if _sha256(implementation_path.read_bytes()) != implementation["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    for source in contract["sources"]:
        source_path = _root_path(source["path"])
        if _sha256(source_path.read_bytes()) != source["file_sha256"]:
            raise RuntimeError(f"source hash mismatch: {source_path.name}")


def _position(rates: dict[str, Decimal]) -> tuple[str, str, dict[str, Decimal]]:
    long_asset = min(rates, key=lambda asset: (rates[asset], asset))
    short_asset = max(rates, key=lambda asset: (rates[asset], asset))
    if long_asset == short_asset:
        raise RuntimeError("funding dispersion selection collapsed")
    return long_asset, short_asset, {long_asset: Decimal(1), short_asset: Decimal(-1)}


def _drawdown(increments: list[Decimal]) -> Decimal:
    cumulative = ZERO
    peak = ZERO
    maximum = ZERO
    for increment in increments:
        cumulative += increment
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _positive_block_concentration(
    increments: list[Decimal], block_size: int
) -> Decimal:
    totals = [
        sum(increments[start : start + block_size], ZERO)
        for start in range(0, len(increments), block_size)
    ]
    positives = [value for value in totals if value > 0]
    if not positives:
        return ZERO
    return max(positives) / sum(positives, ZERO)


def _bootstrap_lower_bound(
    increments: list[Decimal], *, block_size: int, resamples: int, seed: int
) -> Decimal:
    rng = random.Random(seed)
    n = len(increments)
    starts = list(range(n))
    totals: list[Decimal] = []
    for _ in range(resamples):
        sample: list[Decimal] = []
        while len(sample) < n:
            start = rng.choice(starts)
            sample.extend(
                increments[(start + offset) % n] for offset in range(block_size)
            )
        totals.append(sum(sample[:n], ZERO))
    totals.sort()
    # Bonferroni-adjusted one-sided 95% lower tail across three frozen roles.
    index = max(0, int(resamples * (0.05 / 3)) - 1)
    return totals[index]


def _role_result(
    intervals: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    execution_bips = Decimal(contract["economics"]["one_way_execution_bips"])
    per_leg_hurdle = Decimal(
        contract["economics"]["annual_capital_hurdle_percent_per_leg"]
    )
    interval_hours = Decimal(contract["population"]["interval_hours"])
    capital_per_interval = (
        Decimal(2)
        * per_leg_hurdle
        / Decimal(100)
        * interval_hours
        / Decimal(24)
        / Decimal(365)
        * TEN_THOUSAND
    )
    previous: dict[str, Decimal] = {}
    increments: list[Decimal] = []
    gross_funding = ZERO
    execution = ZERO
    pair_counts: Counter[str] = Counter()
    oracle_match_count = 0

    for interval in intervals:
        long_asset, short_asset, position = _position(interval["lagged_rates"])
        oracle_long, oracle_short, _ = _position(interval["realized_rates"])
        oracle_match_count += (long_asset, short_asset) == (oracle_long, oracle_short)
        pair_counts[f"long_{long_asset}_short_{short_asset}"] += 1
        turnover = sum(
            abs(position.get(asset, ZERO) - previous.get(asset, ZERO))
            for asset in set(position) | set(previous)
        )
        interval_execution = turnover * execution_bips
        interval_funding = (
            interval["realized_rates"][short_asset]
            - interval["realized_rates"][long_asset]
        ) * TEN_THOUSAND
        increments.append(interval_funding - interval_execution - capital_per_interval)
        gross_funding += interval_funding
        execution += interval_execution
        previous = position

    close_cost = sum(abs(value) for value in previous.values()) * execution_bips
    execution += close_cost
    increments[-1] -= close_cost
    capital = capital_per_interval * Decimal(len(intervals))
    net = sum(increments, ZERO)
    bootstrap = _bootstrap_lower_bound(
        increments,
        block_size=contract["bootstrap"]["block_intervals"],
        resamples=contract["bootstrap"]["resamples"],
        seed=seed,
    )
    concentration = _positive_block_concentration(
        increments, contract["bootstrap"]["block_intervals"]
    )
    maximum_drawdown = _drawdown(increments)
    passes = (
        net > 0
        and bootstrap > 0
        and concentration
        <= Decimal(contract["gates"]["maximum_positive_week_concentration"])
        and maximum_drawdown <= Decimal(contract["gates"]["maximum_net_drawdown_bips"])
    )
    return {
        "interval_count": len(intervals),
        "duration_days": str(Decimal(len(intervals)) * interval_hours / Decimal(24)),
        "gross_causal_funding_bips": str(gross_funding),
        "turnover_execution_bips": str(execution),
        "two_leg_capital_hurdle_bips": str(capital),
        "net_after_frozen_hurdles_bips": str(net),
        "family_adjusted_block_bootstrap_lower_bound_bips": str(bootstrap),
        "maximum_net_drawdown_bips": str(maximum_drawdown),
        "maximum_positive_week_concentration": str(concentration),
        "oracle_pair_match_fraction": str(
            Decimal(oracle_match_count) / Decimal(len(intervals))
        ),
        "causal_pair_counts": dict(sorted(pair_counts.items())),
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    _validate_contract(contract, contract_path)

    series: dict[str, list[dict[str, Any]]] = {}
    for source in contract["sources"]:
        rows = _load_rows(_root_path(source["path"]))
        if len(rows) != contract["population"]["source_row_count"]:
            raise RuntimeError("source row count changed")
        if any(row.get("symbol") != source["symbol"] for row in rows):
            raise RuntimeError("source symbol changed")
        series[source["asset"]] = sorted(rows, key=lambda row: row["fundingTime"])

    assets = contract["population"]["assets"]
    intervals: list[dict[str, Any]] = []
    maximum_skew = 0
    expected_step = contract["population"]["interval_hours"] * 60 * 60 * 1000
    for index in range(contract["population"]["source_row_count"]):
        times = [int(series[asset][index]["fundingTime"]) for asset in assets]
        maximum_skew = max(maximum_skew, max(times) - min(times))
        if index and any(
            abs(
                int(series[asset][index]["fundingTime"])
                - int(series[asset][index - 1]["fundingTime"])
                - expected_step
            )
            > contract["population"]["maximum_schedule_jitter_ms"]
            for asset in assets
        ):
            raise RuntimeError("funding schedule continuity failed")
        if index == 0:
            continue
        intervals.append(
            {
                "funding_time_ms": max(times),
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
    if maximum_skew > contract["population"]["maximum_cross_asset_skew_ms"]:
        raise RuntimeError("cross-asset funding timestamp skew failed")
    if len(intervals) != contract["population"]["decision_interval_count"]:
        raise RuntimeError("decision interval count changed")

    role_results: dict[str, Any] = {}
    oracle_results: dict[str, Any] = {}
    for role_index, (role, bounds) in enumerate(contract["roles"].items()):
        subset = intervals[bounds["start"] : bounds["stop"]]
        role_results[role] = _role_result(
            subset, contract, seed=contract["bootstrap"]["seed"] + role_index
        )
        oracle_gross = sum(
            (
                max(interval["realized_rates"].values())
                - min(interval["realized_rates"].values())
            )
            * TEN_THOUSAND
            for interval in subset
        )
        capital = Decimal(role_results[role]["two_leg_capital_hurdle_bips"])
        oracle_results[role] = {
            "perfect_foresight_zero_execution_gross_funding_bips": str(oracle_gross),
            "perfect_foresight_zero_execution_net_after_two_leg_capital_bips": str(
                oracle_gross - capital
            ),
            "clears_two_leg_capital_hurdle": oracle_gross > capital,
        }

    oracle_all_roles = all(
        row["clears_two_leg_capital_hurdle"] for row in oracle_results.values()
    )
    causal_all_roles = all(row["passes"] for row in role_results.values())
    price_capture_justified = oracle_all_roles and causal_all_roles
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "population": {
            "assets": assets,
            "source_rows_per_asset": contract["population"]["source_row_count"],
            "decision_intervals": len(intervals),
            "maximum_cross_asset_funding_time_skew_ms": maximum_skew,
            "network_requests": 0,
        },
        "perfect_foresight_dominance_bound": oracle_results,
        "causal_lagged_dispersion_strategy": role_results,
        "gates": {
            "perfect_foresight_clears_capital_in_every_role": oracle_all_roles,
            "causal_strategy_passes_every_role": causal_all_roles,
            "price_capture_justified": price_capture_justified,
        },
        "adjudication": {
            "accepted_edge": False,
            "stable_profitability_proved": False,
            "deployment_ready": False,
            "public_profit_floor_quote_units": "0",
            "status": (
                "retained_funding_gate_passed_freeze_separate_price_regime_test"
                if price_capture_justified
                else "rejected_before_price_requests"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output = _root_path(contract["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "causal_roles_passed": sum(
                    row["passes"] for row in role_results.values()
                ),
                "oracle_roles_clearing_capital": sum(
                    row["clears_two_leg_capital_hurdle"]
                    for row in oracle_results.values()
                ),
                "payloads_printed": 0,
                "price_capture_justified": price_capture_justified,
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
