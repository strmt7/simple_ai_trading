"""Adjudicate retained Aster USD1 versus Binance USDT funding histories."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)


SCHEMA = "binance-wallet-aster-usd1-funding-prefilter-v1"
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
        for venue in ("aster", "binance"):
            item = source[venue]
            source_path = _root_path(item["path"])
            if _sha256(source_path.read_bytes()) != item["file_sha256"]:
                raise RuntimeError(f"source hash mismatch: {source_path.name}")
    for item in contract["lineage_sources"]:
        source_path = _root_path(item["path"])
        if _sha256(source_path.read_bytes()) != item["file_sha256"]:
            raise RuntimeError(f"lineage source hash mismatch: {source_path.name}")


def _normalized_series(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    interval_ms: int,
    maximum_jitter_ms: int,
) -> tuple[dict[int, Decimal], int]:
    result: dict[int, Decimal] = {}
    maximum_jitter = 0
    for row in rows:
        if row.get("symbol") != symbol:
            raise RuntimeError(f"source symbol changed for {symbol}")
        timestamp = int(row["fundingTime"])
        normalized = ((timestamp + interval_ms // 2) // interval_ms) * interval_ms
        jitter = abs(timestamp - normalized)
        maximum_jitter = max(maximum_jitter, jitter)
        if jitter > maximum_jitter_ms:
            raise RuntimeError(f"funding timestamp jitter exceeded for {symbol}")
        if normalized in result:
            raise RuntimeError(f"duplicate normalized funding time for {symbol}")
        result[normalized] = Decimal(str(row["fundingRate"]))
    return result, maximum_jitter


def _role_result(spreads: list[Decimal], contract: dict[str, Any]) -> dict[str, Any]:
    count = len(spreads)
    if count == 0:
        raise RuntimeError("empty role")
    interval_hours = Decimal(contract["population"]["interval_hours"])
    duration_days = Decimal(count) * interval_hours / Decimal(24)
    gross_bips = sum(spreads, ZERO) * TEN_THOUSAND
    execution_bips = Decimal(contract["economics"]["round_trip_execution_bips"])
    annual_capital_percent = Decimal(
        contract["economics"]["annual_two_leg_capital_hurdle_percent"]
    )
    capital_bips = (
        annual_capital_percent
        / Decimal(100)
        * duration_days
        / Decimal(365)
        * TEN_THOUSAND
    )
    fx_stress_bips = Decimal(contract["economics"]["usd1_usdt_stress_bips"])
    net_before_fx = gross_bips - execution_bips - capital_bips
    net_after_fx = net_before_fx - fx_stress_bips
    positive_fraction = Decimal(sum(value > ZERO for value in spreads)) / Decimal(count)
    split = count // 2
    first_half = sum(spreads[:split], ZERO) * TEN_THOUSAND
    second_half = sum(spreads[split:], ZERO) * TEN_THOUSAND
    passes = (
        net_after_fx > ZERO
        and positive_fraction
        >= Decimal(contract["gates"]["minimum_positive_interval_fraction"])
        and first_half > ZERO
        and second_half > ZERO
    )
    return {
        "row_count": count,
        "duration_days": str(duration_days),
        "gross_funding_spread_bips": str(gross_bips),
        "round_trip_execution_bips": str(execution_bips),
        "two_leg_capital_hurdle_bips": str(capital_bips),
        "usd1_usdt_stress_bips": str(fx_stress_bips),
        "net_before_fx_stress_bips": str(net_before_fx),
        "net_after_all_frozen_hurdles_bips": str(net_after_fx),
        "positive_interval_fraction": str(positive_fraction),
        "first_half_gross_bips": str(first_half),
        "second_half_gross_bips": str(second_half),
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    _validate_contract(contract, contract_path)

    population = contract["population"]
    interval_ms = int(population["interval_hours"]) * 60 * 60 * 1000
    aligned_by_asset: dict[str, list[tuple[int, Decimal, Decimal]]] = {}
    source_summaries: list[dict[str, Any]] = []
    for source in contract["sources"]:
        venue_series: dict[str, dict[int, Decimal]] = {}
        venue_jitter: dict[str, int] = {}
        for venue in ("aster", "binance"):
            item = source[venue]
            series, jitter = _normalized_series(
                _load_rows(_root_path(item["path"])),
                symbol=item["symbol"],
                interval_ms=interval_ms,
                maximum_jitter_ms=population["maximum_schedule_jitter_ms"],
            )
            venue_series[venue] = series
            venue_jitter[venue] = jitter
        common = sorted(set(venue_series["aster"]) & set(venue_series["binance"]))
        if len(common) != population["aligned_row_count"]:
            raise RuntimeError(f"aligned row count changed for {source['asset']}")
        if common[0] != population["first_funding_bucket_ms"]:
            raise RuntimeError("first funding bucket changed")
        if common[-1] != population["last_funding_bucket_ms"]:
            raise RuntimeError("last funding bucket changed")
        if any(right - left != interval_ms for left, right in zip(common, common[1:])):
            raise RuntimeError("aligned funding schedule is incomplete")
        aligned_by_asset[source["asset"]] = [
            (time, venue_series["aster"][time], venue_series["binance"][time])
            for time in common
        ]
        source_summaries.append(
            {
                "asset": source["asset"],
                "aligned_row_count": len(common),
                "aster_maximum_schedule_jitter_ms": venue_jitter["aster"],
                "binance_maximum_schedule_jitter_ms": venue_jitter["binance"],
            }
        )

    role_results: dict[str, dict[str, Any]] = {}
    survivors: list[str] = []
    for asset in population["assets"]:
        rows = aligned_by_asset[asset]
        training = contract["roles"]["training"]
        training_rows = rows[training["start"] : training["stop"]]
        training_delta = sum(
            (aster - binance for _, aster, binance in training_rows), ZERO
        )
        if training_delta >= ZERO:
            orientation = "short_aster_USD1_long_binance_USDT"
            sign = Decimal(1)
        else:
            orientation = "long_aster_USD1_short_binance_USDT"
            sign = Decimal(-1)
        asset_roles: dict[str, Any] = {}
        for role, bounds in contract["roles"].items():
            subset = rows[bounds["start"] : bounds["stop"]]
            spreads = [(aster - binance) * sign for _, aster, binance in subset]
            asset_roles[role] = _role_result(spreads, contract)
        passes = all(result["passes"] for result in asset_roles.values())
        if passes:
            survivors.append(asset)
        role_results[asset] = {
            "orientation_selected_from_training_only": orientation,
            "roles": asset_roles,
            "passes_every_role": passes,
        }

    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "population": {
            "assets": population["assets"],
            "aligned_row_count_per_asset": population["aligned_row_count"],
            "source_summaries": source_summaries,
        },
        "asset_results": role_results,
        "gates": {
            "surviving_assets": survivors,
            "funding_only_prefilter_passed": bool(survivors),
            "basis_or_book_capture_justified": bool(survivors),
        },
        "adjudication": {
            "status": (
                "funding_prefilter_survivors_require_basis_and_depth"
                if survivors
                else "terminal_funding_prefilter_rejection_before_books"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "stable_profitability_proved": False,
            "public_profit_floor_quote_units": "0",
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    output["result_sha256"] = _canonical_hash(output, "result_sha256")
    output_path = _root_path(contract["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "aligned_rows_per_asset": population["aligned_row_count"],
                "surviving_assets": survivors,
                "basis_or_book_capture_justified": bool(survivors),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
