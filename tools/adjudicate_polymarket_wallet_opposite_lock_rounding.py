"""Conservatively upper-bound fee-rounding drag on frozen opposite locks."""

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


getcontext().prec = 50
SCHEMA = "polymarket-wallet-opposite-lock-rounding-correction-v1"
FEE_RATE = Decimal("0.07")
FEE_QUANTUM = Decimal("0.00001")
ADVERSE_TICK = Decimal("0.01")


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


def _all_in(price: Decimal) -> Decimal:
    return price + FEE_RATE * price * (Decimal(1) - price)


def _price_from_all_in(cost: Decimal) -> Decimal:
    """Invert the strictly increasing published all-in fee curve."""
    low = Decimal(0)
    high = Decimal(1)
    for _ in range(220):
        midpoint = (low + high) / 2
        if _all_in(midpoint) < cost:
            low = midpoint
        else:
            high = midpoint
    price = (low + high) / 2
    if abs(_all_in(price) - cost) > Decimal("1e-40"):
        raise RuntimeError("could not reconstruct first-leg price")
    return price


def _ceiling_rounding_drag(quantity: Decimal, price: Decimal) -> Decimal:
    continuous = quantity * FEE_RATE * price * (Decimal(1) - price)
    ceiling_fee = continuous.quantize(FEE_QUANTUM, rounding=ROUND_CEILING)
    return ceiling_fee - continuous


def _correct(base: dict[str, Any], base_path: Path, fees_path: Path) -> dict[str, Any]:
    if _canonical_hash(base, "result_sha256") != base.get("result_sha256"):
        raise RuntimeError("base result self-hash mismatch")
    locks = base.get("analysis", {}).get("locks")
    if not isinstance(locks, list) or not locks:
        raise RuntimeError("base result has no locks")

    corrected: list[dict[str, str]] = []
    total_base = Decimal(0)
    total_drag = Decimal(0)
    total_corrected = Decimal(0)
    for index, lock in enumerate(locks):
        if not isinstance(lock, dict):
            raise RuntimeError(f"lock {index} is not an object")
        quantity = _decimal(lock["matched_shares"], "matched shares")
        first_cost = _decimal(
            lock["first_leg_all_in_cost_per_share"], "first all-in cost"
        )
        first_price = _price_from_all_in(first_cost)
        hedge_price = min(
            Decimal(1),
            _decimal(lock["hedge_observed_price"], "hedge price") + ADVERSE_TICK,
        )
        base_pnl = _decimal(lock["locked_pnl"], "base locked pnl")
        first_drag = _ceiling_rounding_drag(quantity, first_price)
        hedge_drag = _ceiling_rounding_drag(quantity, hedge_price)
        drag = first_drag + hedge_drag
        corrected_pnl = base_pnl - drag
        corrected.append(
            {
                "base_locked_pnl_pusd": str(base_pnl),
                "condition_id": str(lock["condition_id"]),
                "conservative_fee_rounding_drag_pusd": str(drag),
                "corrected_locked_pnl_pusd": str(corrected_pnl),
                "matched_shares": str(quantity),
            }
        )
        total_base += base_pnl
        total_drag += drag
        total_corrected += corrected_pnl

    nonpositive = [row for row in corrected if Decimal(row["corrected_locked_pnl_pusd"]) <= 0]
    base_total = _decimal(
        base["analysis"]["stress_locked_pnl_pusd"], "base aggregate pnl"
    )
    if total_base != base_total:
        raise RuntimeError("base lock sum does not reconstruct aggregate")
    status = (
        "candidate_survives_conservative_per_matched_fragment_fee_rounding_ceiling"
        if not nonpositive and total_corrected > 0
        else "candidate_fails_conservative_fee_rounding_correction"
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "method": {
            "base_lock_set_reselected": False,
            "fee_curve": "shares * 0.07 * price * (1 - price)",
            "published_fee_quantum_pusd": str(FEE_QUANTUM),
            "rounding_assumption": "ceil_each_leg_of_each_matched_lock_fragment_to_0_00001_pUSD_even_when_the_source_trade_row_was_larger",
            "interpretation": "This is an upper bound on rounding drag at the retained matched-fragment granularity, not a favorable estimate of venue rounding.",
        },
        "source_binding": {
            "base_result_path": str(base_path.relative_to(_root_path("."))).replace(
                "\\", "/"
            ),
            "base_result_sha256": base["result_sha256"],
            "fees_source_path": str(fees_path.relative_to(_root_path("."))).replace(
                "\\", "/"
            ),
            "fees_source_file_sha256": _sha256(fees_path.read_bytes()),
        },
        "analysis": {
            "base_lock_count": len(locks),
            "nonpositive_lock_count_after_correction": len(nonpositive),
            "base_locked_pnl_pusd": str(total_base),
            "conservative_fee_rounding_drag_pusd": str(total_drag),
            "corrected_locked_pnl_pusd": str(total_corrected),
            "minimum_corrected_lock_pnl_pusd": str(
                min(Decimal(row["corrected_locked_pnl_pusd"]) for row in corrected)
            ),
            "locks": corrected,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "historical_out_of_sample_candidate_survives": not nonpositive
            and total_corrected > 0,
            "profitability_claim": False,
            "public_forward_profit_floor_pusd": "0",
            "status": status,
            "trading_authority": False,
        },
        "authority": {
            "account_requests": 0,
            "credentials_used": False,
            "funds_used": False,
            "network_requests": 0,
            "orders_or_transactions": 0,
            "protected_capture_touched": False,
            "signed_requests": 0,
            "trading_authority": False,
        },
    }
    result["implementation"] = {
        "path": "tools/adjudicate_polymarket_wallet_opposite_lock_rounding.py",
        "sha256": _sha256(Path(__file__).read_bytes()),
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--fees-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.base_result.resolve()
    fees_path = args.fees_source.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"output already exists: {output_path.name}")
    result = _correct(_load(base_path), base_path, fees_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result["adjudication"], sort_keys=True))
    print(json.dumps(result["analysis"], sort_keys=True))


if __name__ == "__main__":
    main()
