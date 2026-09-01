"""Exhaustively audit Polymarket US fee rounding from a frozen official source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CENT = Decimal("0.01")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_EVEN)


def _text(value: Decimal) -> str:
    return format(value, "f")


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _validate_source(contract: dict[str, Any]) -> dict[str, Any]:
    outputs = contract["outputs"]
    source_result = json.loads(
        _root_path(outputs["source_result_path"]).read_text(encoding="ascii")
    )
    if _canonical_hash(source_result, "result_sha256") != source_result[
        "result_sha256"
    ]:
        raise RuntimeError("source result hash mismatch")
    if not source_result["source_gate"]["passed"]:
        raise RuntimeError("official fee source gate failed")
    journal = [
        json.loads(line)
        for line in _root_path(outputs["journal_path"])
        .read_text(encoding="ascii")
        .splitlines()
        if line.strip()
    ]
    if len(journal) != 2 or journal[-1].get("status_code") != 200:
        raise RuntimeError("source request journal is not one successful GET")
    raw = _root_path(outputs["raw_path"]).read_bytes()
    if journal[-1].get("response_sha256") != _sha256(raw):
        raise RuntimeError("raw fee source hash differs from request journal")
    return source_result


def _row(
    *, contracts: int, price: Decimal, maker_theta: Decimal, taker_theta: Decimal
) -> dict[str, Any]:
    uncertainty = price * (Decimal(1) - price)
    raw_maker = maker_theta * Decimal(contracts) * uncertainty
    raw_taker = taker_theta * Decimal(contracts) * uncertainty
    rounded_maker = _money(raw_maker)
    rounded_taker = _money(raw_taker)
    uplift = rounded_maker - raw_maker
    effective_fraction = (
        rounded_maker / rounded_taker if rounded_taker > 0 else Decimal(0)
    )
    return {
        "contracts": contracts,
        "price_USD": _text(price),
        "raw_maker_rebate_USD": _text(raw_maker),
        "rounded_maker_rebate_USD": _text(rounded_maker),
        "maker_rounding_uplift_USD": _text(uplift),
        "rounded_taker_fee_USD": _text(rounded_taker),
        "effective_maker_share_of_rounded_taker_fee": _text(effective_fraction),
        "rebate_per_contract_USD": _text(
            rounded_maker / Decimal(contracts)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()

    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract, contract_path)
    source_result = _validate_source(contract)
    output_path = _root_path(contract["outputs"]["adjudication_path"])
    if output_path.exists():
        raise RuntimeError("one-use adjudication output already exists")

    maker_theta = Decimal(contract["fee_identity"]["maker_rebate_theta"])
    taker_theta = Decimal(contract["fee_identity"]["taker_fee_theta"])
    prices = [Decimal(cents) / Decimal(100) for cents in range(1, 100)]
    quantities = range(
        contract["exhaustive_grid"]["minimum_contracts"],
        contract["exhaustive_grid"]["maximum_contracts"] + 1,
    )
    rows = [
        _row(
            contracts=contracts,
            price=price,
            maker_theta=maker_theta,
            taker_theta=taker_theta,
        )
        for price in prices
        for contracts in quantities
    ]
    positive = [
        row for row in rows if Decimal(row["rounded_maker_rebate_USD"]) > 0
    ]
    uplifted = [
        row for row in rows if Decimal(row["maker_rounding_uplift_USD"]) > 0
    ]
    zero_taker_positive_maker = [
        row
        for row in rows
        if Decimal(row["rounded_taker_fee_USD"]) == 0
        and Decimal(row["rounded_maker_rebate_USD"]) > 0
    ]
    maker_exceeds_taker = [
        row
        for row in rows
        if Decimal(row["rounded_maker_rebate_USD"])
        > Decimal(row["rounded_taker_fee_USD"])
    ]
    max_uplift = max(Decimal(row["maker_rounding_uplift_USD"]) for row in rows)
    max_effective = max(
        Decimal(row["effective_maker_share_of_rounded_taker_fee"])
        for row in rows
    )
    max_per_contract = max(
        Decimal(row["rebate_per_contract_USD"]) for row in rows
    )

    def representative(
        field: str, value: Decimal, limit: int = 10
    ) -> list[dict[str, Any]]:
        matches = [row for row in rows if Decimal(row[field]) == value]
        matches.sort(key=lambda row: (row["contracts"], Decimal(row["price_USD"])))
        return matches[:limit]

    source_safe = (
        maker_theta == Decimal("0.0125")
        and taker_theta == Decimal("0.05")
        and source_result["source_gate"]["passed"]
    )
    invariants_pass = (
        not zero_taker_positive_maker
        and not maker_exceeds_taker
        and max_uplift <= Decimal("0.005")
        and max_effective <= Decimal("0.5")
    )
    accepted_scoped = source_safe and invariants_pass and bool(positive)
    result: dict[str, Any] = {
        "schema_version": "polymarket-us-fee-rounding-adjudication-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "source_result": {
            "path": contract["outputs"]["source_result_path"],
            "sha256": source_result["result_sha256"],
            "gate_passed": source_result["source_gate"]["passed"],
        },
        "exhaustive_grid": {
            "price_count": len(prices),
            "quantity_count": len(list(quantities)),
            "combination_count": len(rows),
            "positive_rounded_maker_rebate_count": len(positive),
            "positive_maker_rounding_uplift_count": len(uplifted),
            "zero_taker_fee_positive_maker_rebate_count": len(
                zero_taker_positive_maker
            ),
            "maker_rebate_exceeds_taker_fee_count": len(maker_exceeds_taker),
        },
        "extrema": {
            "maximum_positive_rounding_uplift_per_trade_USD": _text(max_uplift),
            "maximum_effective_maker_share_of_rounded_taker_fee": _text(
                max_effective
            ),
            "maximum_rounded_rebate_per_contract_USD": _text(max_per_contract),
            "maximum_uplift_representative_rows": representative(
                "maker_rounding_uplift_USD", max_uplift
            ),
            "maximum_effective_share_representative_rows": representative(
                "effective_maker_share_of_rounded_taker_fee", max_effective
            ),
        },
        "adjudication": {
            "source_and_invariants_passed": source_safe and invariants_pass,
            "accepted_scoped_structural_edge": accepted_scoped,
            "scope": "exact_positive_USD_maker_rebate_actually_credited_on_an_independently_justified_legitimate_organic_Polymarket_US_maker_fill_after_every_incremental_cost",
            "rounding_uplift_scope": "credit_only_the_actual_owned_fee_receipt_never_an_assumed_per_fill_rounding_frequency",
            "market_direction_forecast_required": False,
            "standalone_profitable_strategy": False,
            "deployment_ready": False,
            "stable_account_qualified_after_all_cost_edge": False,
            "public_forward_profit_floor_USD": "0",
            "next_action": "Only with explicit separate read-only account authority and independently existing bona fide organic maker fills, reconcile exact fill grouping, quantities, prices, rebates, eligibility, adverse selection, inventory, hedge, tax, and every cost. Any order or account mutation requires separate authority.",
        },
        "limitations": contract["limitations"],
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/adjudicate_polymarket_us_fee_rounding.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "accepted_scoped_structural_edge": accepted_scoped,
                "combination_count": len(rows),
                "maximum_effective_maker_share": _text(max_effective),
                "maximum_rounding_uplift_USD": _text(max_uplift),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
