"""Adjudicate frozen Backpack USDC versus Binance USDT funding histories."""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)


SCHEMA = "binance-backpack-cross-venue-funding-prefilter-v1"
ZERO = Decimal(0)
TEN_THOUSAND = Decimal(10_000)
HOUR_MS = 60 * 60 * 1000


def _load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _load_object(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = _load_json(path)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, dict) for row in value)
    ):
        raise RuntimeError(f"{path.name} must contain a nonempty object array")
    return value


def _validate_contract(contract: dict[str, Any], path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    if contract.get("network_requests") != 0:
        raise RuntimeError("network boundary changed")
    if _root_path(contract["output_path"]).exists():
        raise RuntimeError("one-use output already exists")
    implementation = contract["implementation"]
    implementation_path = _root_path(implementation["path"])
    if _sha256(implementation_path.read_bytes()) != implementation["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    preregistration = contract["preregistration"]
    preregistration_path = _root_path(preregistration["path"])
    preregistration_value = _load_object(preregistration_path)
    if (
        _sha256(preregistration_path.read_bytes()) != preregistration["file_sha256"]
        or _canonical_hash(preregistration_value, "contract_sha256")
        != preregistration_value["contract_sha256"]
        or preregistration_value["contract_sha256"]
        != preregistration["canonical_sha256"]
    ):
        raise RuntimeError("preregistration binding changed")
    inventory = contract["inventory"]
    inventory_path = _root_path(inventory["path"])
    if _sha256(inventory_path.read_bytes()) != inventory["file_sha256"]:
        raise RuntimeError("inventory source hash mismatch")
    for source in contract["sources"]:
        for venue in ("backpack", "binance"):
            item = source[venue]
            source_path = _root_path(item["path"])
            if _sha256(source_path.read_bytes()) != item["file_sha256"]:
                raise RuntimeError(f"source hash mismatch: {source_path.name}")
    for item in contract["lineage_sources"]:
        source_path = _root_path(item["path"])
        if _sha256(source_path.read_bytes()) != item["file_sha256"]:
            raise RuntimeError(f"lineage source hash mismatch: {source_path.name}")


def _binance_series(
    rows: list[dict[str, Any]], *, symbol: str, maximum_jitter_ms: int
) -> tuple[dict[int, Decimal], int]:
    interval_ms = 8 * HOUR_MS
    result: dict[int, Decimal] = {}
    maximum_jitter = 0
    for row in rows:
        if row.get("symbol") != symbol:
            raise RuntimeError(f"Binance symbol changed for {symbol}")
        timestamp = int(row["fundingTime"])
        normalized = ((timestamp + interval_ms // 2) // interval_ms) * interval_ms
        jitter = abs(timestamp - normalized)
        maximum_jitter = max(maximum_jitter, jitter)
        if jitter > maximum_jitter_ms or normalized in result:
            raise RuntimeError(f"invalid Binance schedule for {symbol}")
        result[normalized] = Decimal(str(row["fundingRate"]))
    return result, maximum_jitter


def _timestamp_ms(value: Any) -> int:
    text = str(value)
    if text.isdigit():
        numeric = int(text)
        return numeric * 1000 if numeric < 1_000_000_000_000 else numeric
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("Backpack timestamp lacks timezone")
    return int(parsed.timestamp() * 1000)


def _backpack_series(path: Path, *, symbol: str) -> dict[int, Decimal]:
    rows = _load_rows(path)
    result: dict[int, Decimal] = {}
    for row in rows:
        if row.get("symbol") != symbol:
            raise RuntimeError(f"Backpack symbol changed for {symbol}")
        timestamp = _timestamp_ms(row["intervalEndTimestamp"])
        if timestamp % HOUR_MS or timestamp in result:
            raise RuntimeError(f"invalid Backpack hourly schedule for {symbol}")
        result[timestamp] = Decimal(str(row["fundingRate"]))
    ordered = sorted(result)
    if any(right - left != HOUR_MS for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError(f"incomplete Backpack hourly schedule for {symbol}")
    return result


def _validate_inventory(path: Path, *, assets: list[str]) -> None:
    rows = _load_rows(path)
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = row.get("symbol")
        if isinstance(symbol, str):
            if symbol in by_symbol:
                raise RuntimeError(f"duplicate Backpack market: {symbol}")
            by_symbol[symbol] = row
    for asset in assets:
        symbol = f"{asset}_USDC_PERP"
        row = by_symbol.get(symbol)
        if (
            row is None
            or row.get("baseSymbol") != asset
            or row.get("quoteSymbol") != "USDC"
            or row.get("marketType") != "PERP"
            or row.get("orderBookState") != "Open"
            or row.get("visible") is not True
        ):
            raise RuntimeError(f"Backpack instrument identity failed for {symbol}")


def _role_result(spreads: list[Decimal], contract: dict[str, Any]) -> dict[str, Any]:
    count = len(spreads)
    if count == 0:
        raise RuntimeError("empty role")
    duration_days = Decimal(count) / Decimal(3)
    gross_bips = sum(spreads, ZERO) * TEN_THOUSAND
    economics = contract["economics"]
    execution_bips = Decimal(economics["round_trip_execution_bips"])
    capital_bips = (
        Decimal(economics["annual_two_leg_capital_hurdle_percent"])
        / Decimal(100)
        * duration_days
        / Decimal(365)
        * TEN_THOUSAND
    )
    quote_bips = Decimal(economics["usdc_usdt_quote_unit_stress_bips"])
    custody_bips = Decimal(economics["custody_latency_failure_stress_bips"])
    net_bips = gross_bips - execution_bips - capital_bips - quote_bips - custody_bips
    positive_fraction = Decimal(sum(value > ZERO for value in spreads)) / Decimal(count)
    split = count // 2
    first_half = sum(spreads[:split], ZERO) * TEN_THOUSAND
    second_half = sum(spreads[split:], ZERO) * TEN_THOUSAND
    passes = (
        net_bips > ZERO
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
        "usdc_usdt_quote_unit_stress_bips": str(quote_bips),
        "custody_latency_failure_stress_bips": str(custody_bips),
        "net_after_all_frozen_hurdles_bips": str(net_bips),
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
    _validate_inventory(
        _root_path(contract["inventory"]["path"]), assets=population["assets"]
    )
    aligned_by_asset: dict[str, list[tuple[int, Decimal, Decimal]]] = {}
    source_summaries: list[dict[str, Any]] = []
    for source in contract["sources"]:
        backpack = _backpack_series(
            _root_path(source["backpack"]["path"]),
            symbol=source["backpack"]["symbol"],
        )
        binance, jitter = _binance_series(
            _load_rows(_root_path(source["binance"]["path"])),
            symbol=source["binance"]["symbol"],
            maximum_jitter_ms=population["maximum_binance_schedule_jitter_ms"],
        )
        aligned: list[tuple[int, Decimal, Decimal]] = []
        for bucket in range(
            population["first_aligned_bucket_ms"],
            population["last_aligned_bucket_ms"] + 1,
            8 * HOUR_MS,
        ):
            hourly_times = [bucket - offset * HOUR_MS for offset in range(7, -1, -1)]
            if bucket not in binance or any(
                time not in backpack for time in hourly_times
            ):
                raise RuntimeError(f"frozen alignment incomplete for {source['asset']}")
            aligned.append(
                (
                    bucket,
                    sum((backpack[time] for time in hourly_times), ZERO),
                    binance[bucket],
                )
            )
        if len(aligned) != population["aligned_row_count"]:
            raise RuntimeError(f"aligned row count changed for {source['asset']}")
        aligned_by_asset[source["asset"]] = aligned
        source_summaries.append(
            {
                "asset": source["asset"],
                "backpack_hourly_row_count": len(backpack),
                "aligned_row_count": len(aligned),
                "binance_response_row_count": len(binance),
                "binance_maximum_schedule_jitter_ms": jitter,
            }
        )

    asset_results: dict[str, Any] = {}
    survivors: list[str] = []
    for asset in population["assets"]:
        rows = aligned_by_asset[asset]
        training = contract["roles"]["training"]
        training_rows = rows[training["start"] : training["stop"]]
        training_delta = sum(
            (backpack - binance for _, backpack, binance in training_rows), ZERO
        )
        if training_delta >= ZERO:
            orientation = "short_backpack_USDC_long_binance_USDT"
            sign = Decimal(1)
        else:
            orientation = "long_backpack_USDC_short_binance_USDT"
            sign = Decimal(-1)
        role_results: dict[str, Any] = {}
        for role, bounds in contract["roles"].items():
            subset = rows[bounds["start"] : bounds["stop"]]
            spreads = [(backpack - binance) * sign for _, backpack, binance in subset]
            role_results[role] = _role_result(spreads, contract)
        passes = all(result["passes"] for result in role_results.values())
        if passes:
            survivors.append(asset)
        asset_results[asset] = {
            "orientation_selected_from_training_only": orientation,
            "roles": role_results,
            "passes_every_role": passes,
        }

    basis_justified = bool(survivors) and contract["gates"]["promotion_eligible"]
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "preregistration": contract["preregistration"],
        "population": {
            "assets": population["assets"],
            "aligned_row_count_per_asset": population["aligned_row_count"],
            "source_summaries": source_summaries,
        },
        "asset_results": asset_results,
        "gates": {
            "surviving_assets": survivors,
            "funding_only_prefilter_passed": bool(survivors),
            "basis_or_book_capture_justified": basis_justified,
        },
        "adjudication": {
            "status": (
                "funding_prefilter_survivors_require_basis_and_depth"
                if basis_justified
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
                "basis_or_book_capture_justified": basis_justified,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
