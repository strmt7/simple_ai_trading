"""Screen one frozen two-token crypto range/threshold coverage package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from tools import screen_polymarket_exact_two_leg_package as base


SCHEMA = "polymarket-crypto-range-threshold-books-result-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _validate_contract(contract: dict[str, Any], path: Path) -> dict[str, Any]:
    if base._canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    base._frozen_instant(contract.get("frozen_at_utc"))
    if path != base._root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    for implementation in [
        contract["implementation"],
        *contract["dependency_implementations"],
    ]:
        implementation_path = base._root_path(implementation["path"])
        if base._sha256(implementation_path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {implementation_path.name}")
    if contract["execution"]["book_request_count"] != 1:
        raise RuntimeError("exactly one book batch must be frozen")
    if contract["execution"]["maximum_fee_requests"] != 2:
        raise RuntimeError("fee request ceiling must equal two")
    if contract["authority"] != {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests_maximum": 3,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")

    source = contract["prefilter_source"]
    source_path = base._root_path(source["path"])
    raw = source_path.read_bytes()
    if base._sha256(raw) != source["file_sha256"]:
        raise RuntimeError("prefilter file hash mismatch")
    prefilter = json.loads(raw)
    if base._canonical_hash(prefilter, "result_sha256") != source["result_sha256"]:
        raise RuntimeError("prefilter canonical hash mismatch")
    candidates = [
        row
        for row in prefilter["screen"]["packages_ranked_by_displayed_sum"]
        if row["passes_strict_displayed_gross_gate"]
    ]
    if len(candidates) != 1 or candidates[0] != contract["gamma_prefilter"]["package"]:
        raise RuntimeError("frozen candidate differs from complete prefilter")
    if Decimal(candidates[0]["displayed_price_sum_pUSD"]) >= Decimal("1"):
        raise RuntimeError("Gamma candidate does not clear the strict gross gate")
    token_names = contract["package"]["token_names"]
    if len(token_names) != 2 or len(set(token_names)) != 2:
        raise RuntimeError("package must contain exactly two distinct tokens")
    if [contract["tokens"][name]["token_id"] for name in token_names] != [
        candidates[0]["threshold_token_id"],
        *candidates[0]["range_yes_token_ids"],
    ]:
        raise RuntimeError("package token identity changed")
    payouts = [
        sum(Decimal(value) for value in state["payouts"].values())
        for state in contract["payoff_proof"]["states"]
    ]
    if min(payouts) != Decimal(
        contract["payoff_proof"]["optimistic_rule_consistent_floor_pUSD"]
    ):
        raise RuntimeError("payoff states do not prove the frozen optimistic floor")
    return prefilter


def _json_ready(value: Any) -> Any:
    if isinstance(value, Decimal):
        return base._decimal_text(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = base._root_path(args.contract)
    contract = _load(contract_path)
    _validate_contract(contract, contract_path)
    result_path = base._root_path(contract["outputs"]["result_path"])
    journal_path = base._root_path(contract["outputs"]["journal_path"])
    data_root = journal_path.parent
    if data_root.exists() or result_path.exists():
        raise RuntimeError("one-use output already exists")
    (data_root / "raw/fees").mkdir(parents=True)

    token_names = contract["package"]["token_names"]
    token_ids = [contract["tokens"][name]["token_id"] for name in token_names]
    body = json.dumps(
        [{"token_id": token_id} for token_id in token_ids], separators=(",", ":")
    ).encode("ascii")
    books_relative = contract["outputs"]["books_raw_path"]
    books_raw, book_receipt = base._request(
        method="POST",
        url="https://clob.polymarket.com/books",
        body=body,
        name="exact-two-token-book-batch",
        raw_path=base._root_path(books_relative),
        raw_relative_path=books_relative,
        journal_path=journal_path,
    )
    raw_books = json.loads(books_raw)
    books = {str(row["asset_id"]): row for row in raw_books}
    if len(raw_books) != 2 or set(books) != set(token_ids):
        raise RuntimeError("book population differs from contract")

    quantity = Decimal(contract["execution"]["quantity_shares_each_leg"])
    tick_size = Decimal(contract["execution"]["tick_size"])
    timestamps: list[int] = []
    for name in token_names:
        definition = contract["tokens"][name]
        book = books[definition["token_id"]]
        if not (
            str(book["market"]).lower() == definition["condition_id"]
            and bool(book["neg_risk"]) is definition["neg_risk"]
            and Decimal(str(book["min_order_size"])) <= quantity
            and Decimal(str(book["tick_size"])) == tick_size
        ):
            raise RuntimeError(f"book identity changed: {name}")
        timestamps.append(int(book["timestamp"]))
    completed_at_ms = int(book_receipt["completed_at_ms"])
    skew_ms = max(timestamps) - min(timestamps)
    oldest_age_ms = completed_at_ms - min(timestamps)
    newest_age_ms = completed_at_ms - max(timestamps)
    synchronized = skew_ms <= int(
        contract["execution"]["maximum_book_timestamp_skew_ms"]
    )
    source_time_not_future = newest_age_ms >= 0
    fresh = source_time_not_future and oldest_age_ms <= int(
        contract["execution"]["maximum_book_age_ms"]
    )

    zero_fee_fills: dict[str, list[dict[str, Any] | None]] = {}
    for stress_name, ticks in contract["execution"]["stress_ticks"].items():
        zero_fee_fills[stress_name] = [
            base._fill(
                books[contract["tokens"][name]["token_id"]],
                quantity=quantity,
                tick_size=tick_size,
                adverse_ticks=int(ticks),
                fee_model=None,
            )
            for name in token_names
        ]
    floor = quantity * Decimal(
        contract["payoff_proof"]["optimistic_rule_consistent_floor_pUSD"]
    )
    candidate_stress = contract["execution"]["candidate_stress_name"]
    candidate_zero_fee = zero_fee_fills[candidate_stress]
    gross_positive = all(candidate_zero_fee) and floor > sum(
        fill["cost_pUSD"] for fill in candidate_zero_fee if fill is not None
    )

    fee_receipts: dict[str, dict[str, Any]] = {}
    fee_model: PolymarketFeeModel | None = None
    if gross_positive:
        for name in token_names:
            token_id = contract["tokens"][name]["token_id"]
            relative = f"{contract['outputs']['fee_raw_root']}/{token_id}.json"
            raw, receipt = base._request(
                method="GET",
                url=f"https://clob.polymarket.com/fee-rate/{token_id}",
                body=b"",
                name=f"fee-rate-{name}",
                raw_path=base._root_path(relative),
                raw_relative_path=relative,
                journal_path=journal_path,
            )
            if json.loads(raw) != {"base_fee": 1000}:
                raise RuntimeError(f"fee rate changed: {name}")
            fee_receipts[name] = receipt
        schedule = contract["execution"]["fee_schedule"]
        fee_model = PolymarketFeeModel(
            enabled=True,
            rate=Decimal(str(schedule["rate"])),
            exponent=int(schedule["exponent"]),
            taker_only=bool(schedule["takerOnly"]),
        )

    economics: dict[str, Any] = {}
    for stress_name, ticks in contract["execution"]["stress_ticks"].items():
        fills = [
            base._fill(
                books[contract["tokens"][name]["token_id"]],
                quantity=quantity,
                tick_size=tick_size,
                adverse_ticks=int(ticks),
                fee_model=fee_model,
            )
            for name in token_names
        ]
        if not all(fills):
            economics[stress_name] = None
            continue
        cost = sum(fill["cost_pUSD"] for fill in fills if fill is not None)
        fee = (
            sum(fill["fee_pUSD"] for fill in fills if fill is not None)
            if fee_model is not None
            else None
        )
        economics[stress_name] = {
            "adverse_ticks_per_leg": ticks,
            "cost_pUSD": cost,
            "optimistic_zero_fee_profit_floor_pUSD": floor - cost,
            "current_fee_pUSD": fee,
            "after_current_fee_profit_floor_pUSD": (
                floor - cost - fee if fee is not None else None
            ),
            "fills": fills,
        }
    final = economics[candidate_stress]
    passes = bool(
        synchronized
        and fresh
        and final is not None
        and final["after_current_fee_profit_floor_pUSD"] is not None
        and final["after_current_fee_profit_floor_pUSD"] > 0
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "book_receipt": book_receipt,
            "fee_receipts": fee_receipts,
            "book_timestamp_skew_ms": skew_ms,
            "oldest_book_age_at_completion_ms": oldest_age_ms,
            "newest_book_age_at_completion_ms": newest_age_ms,
            "within_frozen_skew_gate": synchronized,
            "within_frozen_age_gate": fresh,
            "source_time_not_future": source_time_not_future,
        },
        "payoff_proof": contract["payoff_proof"],
        "gamma_prefilter": contract["gamma_prefilter"],
        "economics": economics,
        "adjudication": {
            "passes_frozen_candidate_gate": passes,
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "require_independent_recurrence_and_exact_resolution_risk_before_order_capable_work"
                if passes
                else "terminalize_this_exact_pair_without_refetch_or_retry"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    serializable = _json_ready(result)
    serializable["result_sha256"] = base._canonical_hash(
        serializable, "result_sha256"
    )
    result_path.write_text(
        json.dumps(serializable, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "book_request_count": 1,
                "fee_request_count": len(fee_receipts),
                "oldest_book_age_ms": oldest_age_ms,
                "passes_candidate_gate": passes,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
