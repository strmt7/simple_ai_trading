"""Adjudicate retained Binance linear-versus-inverse funding histories offline."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.screen_binance_linear_inverse_funding import (
    ROOT,
    TEN_THOUSAND,
    ZERO,
    _bucket_rows,
    _canonical_hash,
    _sha256,
    _write_json,
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _validate(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != (ROOT / contract["contract_path"]).resolve():
        raise RuntimeError("contract path mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    capture = ROOT / contract["capture_result"]["path"]
    capture_value = _load_object(capture)
    if capture_value.get("result_sha256") != contract["capture_result"]["result_sha256"]:
        raise RuntimeError("capture result identity mismatch")
    if _canonical_hash(capture_value, "result_sha256") != capture_value["result_sha256"]:
        raise RuntimeError("capture result hash mismatch")
    for source in contract["sources"]:
        path = ROOT / source["path"]
        if _sha256(path.read_bytes()) != source["sha256"]:
            raise RuntimeError(f"source hash mismatch: {path.name}")
    if (ROOT / contract["output_path"]).exists():
        raise RuntimeError("one-use adjudication output already exists")


def _aligned_spreads(asset: str, contract: dict[str, Any]) -> list[Decimal]:
    source_map = {row["role"]: row for row in contract["sources"] if row["asset"] == asset}
    coin_rows = json.loads((ROOT / source_map["coin_history"]["path"]).read_bytes())
    linear_rows = json.loads((ROOT / source_map["linear_history"]["path"]).read_bytes())
    interval_ms = int(contract["population"]["funding_interval_hours"]) * 3_600_000
    coin = _bucket_rows(coin_rows, source_map["coin_history"]["symbol"], interval_ms)
    linear = _bucket_rows(
        linear_rows, source_map["linear_history"]["symbol"], interval_ms
    )
    common = sorted(set(coin) & set(linear))
    return [
        (
            Decimal(str(coin[bucket]["fundingRate"]))
            - Decimal(str(linear[bucket]["fundingRate"]))
        )
        * TEN_THOUSAND
        for bucket in common
    ]


def _fixed_combined_out_of_sample(
    spreads: list[Decimal], execution_bips: Decimal
) -> dict[str, Any]:
    training_stop = len(spreads) * 3 // 5
    orientation = Decimal(1) if sum(spreads[:training_stop], ZERO) >= ZERO else Decimal(-1)
    rows = spreads[training_stop:]
    increments = [orientation * value for value in rows]
    gross = sum(increments, ZERO)
    hurdle = execution_bips * Decimal(4)
    return {
        "entry_exit_execution_hurdle_bips": str(hurdle),
        "gross_funding_bips": str(gross),
        "interval_count": len(rows),
        "net_after_one_combined_entry_exit_hurdle_bips": str(gross - hurdle),
        "one_way_break_even_execution_bips": str(gross / Decimal(4)),
        "training_selected_orientation": (
            "short_COIN_M_long_USDS_M"
            if orientation > ZERO
            else "long_COIN_M_short_USDS_M"
        ),
    }


def _lagged_role(spreads: list[Decimal], one_way_fee: Decimal) -> dict[str, Any]:
    gross = ZERO
    turnover = ZERO
    previous: Decimal | None = None
    increments: list[Decimal] = []
    for index in range(1, len(spreads)):
        orientation = Decimal(1) if spreads[index - 1] >= ZERO else Decimal(-1)
        increment = orientation * spreads[index]
        gross += increment
        increments.append(increment)
        if previous is None:
            turnover += Decimal(2)
        elif orientation != previous:
            turnover += Decimal(4)
        previous = orientation
    if previous is not None:
        turnover += Decimal(2)
    execution = turnover * one_way_fee
    return {
        "gross_causal_funding_bips": str(gross),
        "net_after_turnover_bips": str(gross - execution),
        "one_way_execution_bips": str(one_way_fee),
        "turnover_execution_bips": str(execution),
        "turnover_units": str(turnover),
        "zero_cost_positive": gross > ZERO,
        "one_way_break_even_execution_bips": (
            str(gross / turnover) if turnover > ZERO else None
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    _validate(contract, contract_path)
    capture = _load_object(ROOT / contract["capture_result"]["path"])
    capture_pairs = {row["asset"]: row for row in capture["pair_results"]}
    execution = Decimal(contract["economics"]["frozen_one_way_execution_bips"])
    fees = [Decimal(value) for value in contract["economics"]["lagged_fee_sensitivity_bips"]]
    rows: list[dict[str, Any]] = []
    for asset in contract["population"]["assets"]:
        spreads = _aligned_spreads(asset, contract)
        pair = capture_pairs[asset]
        training_stop = len(spreads) * 3 // 5
        validation_stop = len(spreads) * 4 // 5
        role_bounds = {
            "training": (0, training_stop),
            "validation": (training_stop, validation_stop),
            "test": (validation_stop, len(spreads)),
        }
        lagged = {
            role: {
                str(fee): _lagged_role(spreads[start:stop], fee) for fee in fees
            }
            for role, (start, stop) in role_bounds.items()
        }
        fixed = _fixed_combined_out_of_sample(spreads, execution)
        rows.append(
            {
                "asset": asset,
                "fixed_orientation_combined_out_of_sample": fixed,
                "fixed_orientation_observed_months": {
                    "validation": pair["role_results"]["validation"]["observed_months"],
                    "test": pair["role_results"]["test"]["observed_months"],
                },
                "lagged_sign_causal_sensitivity": lagged,
                "terminal_reasons": [
                    "fixed_combined_out_of_sample_net_nonpositive"
                    if Decimal(
                        fixed["net_after_one_combined_entry_exit_hurdle_bips"]
                    )
                    <= ZERO
                    else "fixed_combined_out_of_sample_net_positive",
                    "fixed_orientation_out_of_sample_month_sign_instability",
                    "lagged_sign_turnover_exceeds_gross_at_two_bips_one_way",
                ],
            }
        )
    lagged_two_bip_all_negative = all(
        Decimal(row["lagged_sign_causal_sensitivity"][role]["2"]["net_after_turnover_bips"])
        < ZERO
        for row in rows
        for role in ("training", "validation", "test")
    )
    fixed_all_negative = all(
        Decimal(
            row["fixed_orientation_combined_out_of_sample"][
                "net_after_one_combined_entry_exit_hurdle_bips"
            ]
        )
        < ZERO
        for row in rows
    )
    result: dict[str, Any] = {
        "authority": contract["authority"],
        "capture_result": contract["capture_result"],
        "contract": {
            "contract_sha256": contract["contract_sha256"],
            "path": contract["contract_path"],
        },
        "pair_adjudications": rows,
        "result_sha256": "",
        "schema_version": "binance-linear-inverse-perpetual-funding-adjudication-v1",
        "verdict": {
            "accepted_edge": False,
            "fixed_combined_out_of_sample_all_failed_frozen_hurdle": fixed_all_negative,
            "lagged_sign_all_roles_failed_two_bip_sensitivity": lagged_two_bip_all_negative,
            "profitability_claim": False,
            "status": "terminal_rejected_linear_inverse_perpetual_funding_family",
            "trading_authority": False,
            "retry_trigger": "material COIN-M or USD-M funding cash-flow, collateral, settlement, fee, or contract architecture change",
        },
    }
    if not fixed_all_negative or not lagged_two_bip_all_negative:
        raise RuntimeError("terminal dominance predicates did not hold")
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    _write_json(ROOT / contract["output_path"], result)


if __name__ == "__main__":
    main()
