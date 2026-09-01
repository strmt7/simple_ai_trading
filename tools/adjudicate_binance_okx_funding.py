"""Adjudicate preregistered OKX USDT versus Binance USDT funding histories."""

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


SCHEMA = "binance-okx-cross-venue-funding-prefilter-v1"
ZERO = Decimal(0)
TEN_THOUSAND = Decimal(10_000)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, list) or not value or any(
        not isinstance(row, dict) for row in value
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
    for source in contract["sources"]:
        for venue in ("okx", "binance"):
            item = source[venue]
            source_path = _root_path(item["path"])
            if _sha256(source_path.read_bytes()) != item["file_sha256"]:
                raise RuntimeError(f"source hash mismatch: {source_path.name}")
    for item in contract["lineage_sources"]:
        source_path = _root_path(item["path"])
        if _sha256(source_path.read_bytes()) != item["file_sha256"]:
            raise RuntimeError(f"lineage source hash mismatch: {source_path.name}")


def _validate_inventory(path: Path, contract: dict[str, Any]) -> None:
    payload = _load_object(path)
    rows = payload.get("data")
    if payload.get("code") != "0" or not isinstance(rows, list):
        raise RuntimeError("OKX inventory envelope changed")
    first_bucket = contract["population"]["first_funding_bucket_ms"]
    for source in contract["sources"]:
        matches = [row for row in rows if row.get("instId") == source["okx"]["inst_id"]]
        if len(matches) != 1:
            raise RuntimeError(f"OKX inventory multiplicity changed for {source['asset']}")
        row = matches[0]
        if (
            row.get("instType") != "SWAP"
            or row.get("ctType") != "linear"
            or row.get("settleCcy") != "USDT"
            or row.get("state") != "live"
            or int(row["listTime"]) > first_bucket
        ):
            raise RuntimeError(f"OKX inventory gate failed for {source['asset']}")


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
        if jitter > maximum_jitter_ms or normalized in result:
            raise RuntimeError(f"invalid Binance funding schedule for {symbol}")
        result[normalized] = Decimal(str(row["fundingRate"]))
    return result, maximum_jitter


def _okx_series(
    path: Path,
    *,
    inst_id: str,
    interval_ms: int,
    maximum_jitter_ms: int,
    expected_response_rows: int,
) -> tuple[dict[int, Decimal], int]:
    payload = _load_object(path)
    rows = payload.get("data")
    if (
        payload.get("code") != "0"
        or payload.get("msg") != ""
        or not isinstance(rows, list)
        or len(rows) != expected_response_rows
    ):
        raise RuntimeError(f"OKX funding envelope changed for {inst_id}")
    result: dict[int, Decimal] = {}
    maximum_jitter = 0
    for row in rows:
        if row.get("instId") != inst_id or row.get("instType") != "SWAP":
            raise RuntimeError(f"OKX instrument identity changed for {inst_id}")
        timestamp = int(row["fundingTime"])
        normalized = ((timestamp + interval_ms // 2) // interval_ms) * interval_ms
        jitter = abs(timestamp - normalized)
        maximum_jitter = max(maximum_jitter, jitter)
        if jitter > maximum_jitter_ms or normalized in result:
            raise RuntimeError(f"invalid OKX funding schedule for {inst_id}")
        result[normalized] = Decimal(str(row["realizedRate"]))
    return result, maximum_jitter


def _role_result(spreads: list[Decimal], contract: dict[str, Any]) -> dict[str, Any]:
    count = len(spreads)
    if count == 0:
        raise RuntimeError("empty role")
    interval_hours = Decimal(contract["population"]["interval_hours"])
    duration_days = Decimal(count) * interval_hours / Decimal(24)
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
    quote_bips = Decimal(economics["quote_unit_stress_bips"])
    venue_bips = Decimal(economics["custody_transfer_latency_failure_stress_bips"])
    net_bips = gross_bips - execution_bips - capital_bips - quote_bips - venue_bips
    positive_fraction = Decimal(sum(value > ZERO for value in spreads)) / Decimal(count)
    split = count // 2
    first_half = sum(spreads[:split], ZERO) * TEN_THOUSAND
    second_half = sum(spreads[split:], ZERO) * TEN_THOUSAND
    passes = (
        net_bips > ZERO
        and positive_fraction >= Decimal(contract["gates"]["minimum_positive_interval_fraction"])
        and first_half > ZERO
        and second_half > ZERO
    )
    return {
        "row_count": count,
        "duration_days": str(duration_days),
        "gross_funding_spread_bips": str(gross_bips),
        "round_trip_execution_bips": str(execution_bips),
        "two_leg_capital_hurdle_bips": str(capital_bips),
        "quote_unit_stress_bips": str(quote_bips),
        "custody_transfer_latency_failure_stress_bips": str(venue_bips),
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
    _validate_inventory(_root_path(contract["inventory_path"]), contract)

    population = contract["population"]
    interval_ms = int(population["interval_hours"]) * 60 * 60 * 1000
    aligned_by_asset: dict[str, list[tuple[int, Decimal, Decimal]]] = {}
    source_summaries: list[dict[str, Any]] = []
    for source in contract["sources"]:
        okx, okx_jitter = _okx_series(
            _root_path(source["okx"]["path"]),
            inst_id=source["okx"]["inst_id"],
            interval_ms=interval_ms,
            maximum_jitter_ms=population["maximum_schedule_jitter_ms"],
            expected_response_rows=population["okx_response_row_count"],
        )
        binance, binance_jitter = _normalized_series(
            _load_rows(_root_path(source["binance"]["path"])),
            symbol=source["binance"]["symbol"],
            interval_ms=interval_ms,
            maximum_jitter_ms=population["maximum_schedule_jitter_ms"],
        )
        buckets = list(
            range(
                population["first_funding_bucket_ms"],
                population["last_funding_bucket_ms"] + 1,
                interval_ms,
            )
        )
        if len(buckets) != population["aligned_row_count"] or any(
            bucket not in okx or bucket not in binance for bucket in buckets
        ):
            raise RuntimeError(f"frozen alignment incomplete for {source['asset']}")
        aligned_by_asset[source["asset"]] = [
            (bucket, okx[bucket], binance[bucket]) for bucket in buckets
        ]
        source_summaries.append(
            {
                "asset": source["asset"],
                "okx_response_row_count": len(okx),
                "aligned_row_count": len(buckets),
                "okx_maximum_schedule_jitter_ms": okx_jitter,
                "binance_maximum_schedule_jitter_ms": binance_jitter,
            }
        )

    asset_results: dict[str, Any] = {}
    survivors: list[str] = []
    for asset in population["assets"]:
        rows = aligned_by_asset[asset]
        training = contract["roles"]["training"]
        training_rows = rows[training["start"] : training["stop"]]
        training_delta = sum((okx - binance for _, okx, binance in training_rows), ZERO)
        if training_delta >= ZERO:
            orientation = "short_OKX_USDT_long_Binance_USDT"
            sign = Decimal(1)
        else:
            orientation = "long_OKX_USDT_short_Binance_USDT"
            sign = Decimal(-1)
        role_results: dict[str, Any] = {}
        for role, bounds in contract["roles"].items():
            subset = rows[bounds["start"] : bounds["stop"]]
            spreads = [(okx - binance) * sign for _, okx, binance in subset]
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
        "contract": {"path": contract["contract_path"], "sha256": contract["contract_sha256"]},
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
        json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
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
