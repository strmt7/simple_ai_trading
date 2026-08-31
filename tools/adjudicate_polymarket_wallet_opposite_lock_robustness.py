"""Stress the frozen opposite-lock set without reselecting any historical row."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, getcontext
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.adjudicate_polymarket_wallet_opposite_lock_rounding import (
    _price_from_all_in,
)


getcontext().prec = 50
SCHEMA = "polymarket-wallet-opposite-lock-robustness-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise RuntimeError(f"non-finite {label}")
    return parsed


def _ceil_fee(
    quantity: Decimal, price: Decimal, fee_rate: Decimal, fee_quantum: Decimal
) -> Decimal:
    continuous = quantity * fee_rate * price * (Decimal(1) - price)
    return continuous.quantize(fee_quantum, rounding=ROUND_CEILING)


def _rounded_lock_pnl(
    lock: dict[str, Any],
    *,
    adverse_ticks: int,
    tick_size: Decimal,
    fee_rate: Decimal,
    fee_quantum: Decimal,
) -> Decimal:
    quantity = _decimal(lock["matched_shares"], "matched shares")
    first_all_in = _decimal(
        lock["first_leg_all_in_cost_per_share"], "first all-in cost"
    )
    first_price = _price_from_all_in(first_all_in)
    hedge_price = min(
        Decimal(1),
        _decimal(lock["hedge_observed_price"], "hedge observed price")
        + tick_size * adverse_ticks,
    )
    first_cost = quantity * first_price + _ceil_fee(
        quantity, first_price, fee_rate, fee_quantum
    )
    hedge_cost = quantity * hedge_price + _ceil_fee(
        quantity, hedge_price, fee_rate, fee_quantum
    )
    return quantity - first_cost - hedge_cost


def _percentile(values: list[Decimal], numerator: int, denominator: int) -> Decimal:
    """Return a deterministic nearest-rank percentile from a nonempty list."""
    ordered = sorted(values)
    rank = (len(ordered) * numerator + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract self-hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an explicit UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")
    expected_authority = {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "network_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "signed_requests": 0,
        "trading_authority": False,
    }
    if contract.get("authority") != expected_authority:
        raise RuntimeError("authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")
    for source in contract["source_bindings"]:
        path = _root_path(source["path"])
        if _sha256(path.read_bytes()) != source["file_sha256"]:
            raise RuntimeError(f"source hash mismatch: {path.name}")


def _adjudicate(contract: dict[str, Any]) -> dict[str, Any]:
    sources = {source["role"]: source for source in contract["source_bindings"]}
    base_path = _root_path(sources["frozen_validation_result"]["path"])
    correction_path = _root_path(sources["fee_rounding_correction"]["path"])
    base = _load(base_path)
    correction = _load(correction_path)
    if _canonical_hash(base, "result_sha256") != base.get("result_sha256"):
        raise RuntimeError("base result self-hash mismatch")
    if _canonical_hash(correction, "result_sha256") != correction.get("result_sha256"):
        raise RuntimeError("rounding correction self-hash mismatch")
    if base["result_sha256"] != sources["frozen_validation_result"]["self_sha256"]:
        raise RuntimeError("base result canonical hash mismatch")
    if correction["result_sha256"] != sources["fee_rounding_correction"]["self_sha256"]:
        raise RuntimeError("rounding correction canonical hash mismatch")

    locks = base.get("analysis", {}).get("locks")
    if (
        not isinstance(locks, list)
        or len(locks) != contract["fixed_population"]["lock_count"]
    ):
        raise RuntimeError("fixed lock population changed")
    if (
        base["analysis"]["matched_shares"]
        != contract["fixed_population"]["matched_shares"]
    ):
        raise RuntimeError("fixed matched-share population changed")

    stress = contract["stress_grid"]
    tick_size = _decimal(stress["tick_size_pusd"], "tick size")
    fee_rate = _decimal(stress["fee_rate"], "fee rate")
    fee_quantum = _decimal(stress["fee_quantum_pusd"], "fee quantum")
    adverse_ticks = [int(value) for value in stress["total_hedge_adverse_ticks"]]
    fixed_costs = [
        _decimal(value, "fixed per-lock cost")
        for value in stress["fixed_operating_cost_per_lock_pusd"]
    ]
    if adverse_ticks != sorted(set(adverse_ticks)) or adverse_ticks[0] != 1:
        raise RuntimeError("adverse-tick grid must be unique, sorted, and start at one")
    if fixed_costs != sorted(set(fixed_costs)) or fixed_costs[0] != 0:
        raise RuntimeError("fixed-cost grid must be unique, sorted, and start at zero")

    baseline_pnls = [
        _rounded_lock_pnl(
            lock,
            adverse_ticks=1,
            tick_size=tick_size,
            fee_rate=fee_rate,
            fee_quantum=fee_quantum,
        )
        for lock in locks
    ]
    corrected_rows = correction["analysis"]["locks"]
    if len(corrected_rows) != len(baseline_pnls):
        raise RuntimeError("rounding-correction lock population changed")
    for index, (computed, retained) in enumerate(zip(baseline_pnls, corrected_rows)):
        expected = _decimal(retained["corrected_locked_pnl_pusd"], "corrected pnl")
        if computed != expected:
            raise RuntimeError(f"one-tick baseline mismatch at lock {index}")

    scenarios: list[dict[str, Any]] = []
    gate_scenario: dict[str, Any] | None = None
    gate = contract["robustness_gate"]
    for ticks in adverse_ticks:
        stressed = [
            _rounded_lock_pnl(
                lock,
                adverse_ticks=ticks,
                tick_size=tick_size,
                fee_rate=fee_rate,
                fee_quantum=fee_quantum,
            )
            for lock in locks
        ]
        for fixed_cost in fixed_costs:
            net = [value - fixed_cost for value in stressed]
            positive_indices = [index for index, value in enumerate(net) if value > 0]
            asset_totals: dict[str, Decimal] = {}
            for asset in sorted({str(lock["asset"]) for lock in locks}):
                asset_totals[asset] = sum(
                    (
                        net[index]
                        for index, lock in enumerate(locks)
                        if str(lock["asset"]) == asset
                    ),
                    Decimal(0),
                )
            row = {
                "all_asset_aggregate_pnl_positive": all(
                    value > 0 for value in asset_totals.values()
                ),
                "asset_aggregate_pnl_pusd": {
                    asset: str(value) for asset, value in asset_totals.items()
                },
                "fixed_operating_cost_per_lock_pusd": str(fixed_cost),
                "positive_lock_count": len(positive_indices),
                "positive_lock_fraction": str(
                    Decimal(len(positive_indices)) / Decimal(len(locks))
                ),
                "positive_lock_matched_shares": str(
                    sum(
                        (
                            _decimal(locks[index]["matched_shares"], "matched shares")
                            for index in positive_indices
                        ),
                        Decimal(0),
                    )
                ),
                "total_hedge_adverse_ticks": ticks,
                "total_locked_pnl_pusd": str(sum(net, Decimal(0))),
            }
            scenarios.append(row)
            if ticks == int(
                gate["total_hedge_adverse_ticks"]
            ) and fixed_cost == _decimal(
                gate["fixed_operating_cost_per_lock_pusd"], "gate fixed cost"
            ):
                gate_scenario = row
    if gate_scenario is None:
        raise RuntimeError("predeclared gate scenario is absent from stress grid")

    maximum_extra_ticks: list[int] = []
    search_ceiling = int(stress["break_even_tick_search_ceiling"])
    for lock in locks:
        maximum_total = 0
        for ticks in range(1, search_ceiling + 1):
            pnl = _rounded_lock_pnl(
                lock,
                adverse_ticks=ticks,
                tick_size=tick_size,
                fee_rate=fee_rate,
                fee_quantum=fee_quantum,
            )
            if pnl > 0:
                maximum_total = ticks
            else:
                break
        maximum_extra_ticks.append(maximum_total - 1)

    gate_passed = (
        _decimal(gate_scenario["total_locked_pnl_pusd"], "gate total pnl") > 0
        and _decimal(gate_scenario["positive_lock_fraction"], "positive fraction")
        >= _decimal(gate["minimum_positive_lock_fraction"], "gate fraction")
        and gate_scenario["all_asset_aggregate_pnl_positive"] is True
    )
    status = (
        "fixed_historical_lock_set_survives_predeclared_robustness_gate"
        if gate_passed
        else "fixed_historical_lock_set_fails_predeclared_robustness_gate"
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": {
            "base_lock_set_reselected": False,
            "fee_rounding": "ceil_each_leg_of_each_fixed_matched_fragment_to_0_00001_pUSD",
            "fixed_cost_application": "once_per_retained_lock_even_when_stressed_pnl_is_nonpositive",
            "interpretation": "A zero-network robustness profile of the already-selected OOS locks, not evidence of current executable prices or first-leg justification.",
            "stress_grid": stress,
        },
        "source_binding": {
            source["role"]: {
                "file_sha256": source["file_sha256"],
                "path": source["path"],
                **(
                    {"self_sha256": source["self_sha256"]}
                    if "self_sha256" in source
                    else {}
                ),
            }
            for source in contract["source_bindings"]
        },
        "analysis": {
            "baseline_one_tick_break_even_fixed_cost_per_lock_pusd": {
                "maximum": str(max(baseline_pnls)),
                "median_nearest_rank": str(_percentile(baseline_pnls, 1, 2)),
                "minimum": str(min(baseline_pnls)),
                "p10_nearest_rank": str(_percentile(baseline_pnls, 1, 10)),
                "p90_nearest_rank": str(_percentile(baseline_pnls, 9, 10)),
            },
            "base_lock_count": len(locks),
            "base_matched_shares": base["analysis"]["matched_shares"],
            "maximum_additional_whole_adverse_ticks_above_baseline": {
                "maximum": max(maximum_extra_ticks),
                "median_nearest_rank": int(
                    _percentile([Decimal(value) for value in maximum_extra_ticks], 1, 2)
                ),
                "minimum": min(maximum_extra_ticks),
                "p10_nearest_rank": int(
                    _percentile(
                        [Decimal(value) for value in maximum_extra_ticks], 1, 10
                    )
                ),
                "p90_nearest_rank": int(
                    _percentile(
                        [Decimal(value) for value in maximum_extra_ticks], 9, 10
                    )
                ),
                "search_ceiling_total_ticks": search_ceiling,
            },
            "predeclared_gate_passed": gate_passed,
            "predeclared_gate_scenario": gate_scenario,
            "scenarios": scenarios,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "historical_out_of_sample_robustness_gate_passed": gate_passed,
            "profitability_claim": False,
            "public_forward_profit_floor_pusd": "0",
            "status": status,
            "trading_authority": False,
        },
        "authority": contract["authority"],
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
    }
    result["implementation"] = {
        "path": "tools/adjudicate_polymarket_wallet_opposite_lock_robustness.py",
        "sha256": _sha256(Path(__file__).read_bytes()),
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"output already exists: {output_path.name}")
    contract = _load(contract_path)
    _validate_contract(contract, contract_path)
    result = _adjudicate(contract)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result["adjudication"], sort_keys=True))
    print(json.dumps(result["analysis"]["predeclared_gate_scenario"], sort_keys=True))


if __name__ == "__main__":
    main()
