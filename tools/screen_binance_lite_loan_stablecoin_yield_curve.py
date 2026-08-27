"""One-use public screen for Binance Lite Loan funded stablecoin bonus yield."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal, getcontext
from pathlib import Path
from typing import Any

getcontext().prec = 50


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _floor_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def _read_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), _sha256_bytes(raw)


def _capture_once(
    url: str, raw_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    requested_before_utc = _utc_now()
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-research/1",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
        status = response.status
        raw = response.read()
    received_after_utc = _utc_now()

    response_path = raw_dir / "response-001.json"
    response_path.write_bytes(raw)
    response_sha256 = _sha256_bytes(raw)
    journal = {
        "method": "GET",
        "received_after_utc": received_after_utc,
        "request_index": 1,
        "requested_before_utc": requested_before_utc,
        "response_bytes": len(raw),
        "response_path": response_path.as_posix(),
        "response_sha256": response_sha256,
        "status": status,
        "url": url,
    }
    journal_path = raw_dir / "journal.json"
    journal_path.write_bytes(_canonical_bytes(journal) + b"\n")

    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("book ticker response must be a list")
    return payload, {
        "journal_path": journal_path.as_posix(),
        "journal_sha256": _sha256_bytes(journal_path.read_bytes()),
        "request_count": 1,
        "response_bytes": len(raw),
        "response_sha256": response_sha256,
        "requested_before_utc": requested_before_utc,
        "received_after_utc": received_after_utc,
    }


def _books_by_symbol(
    rows: list[dict[str, Any]], symbols: list[str]
) -> dict[str, dict[str, Decimal]]:
    if len(rows) != len(symbols):
        raise ValueError("book ticker response count does not match frozen symbols")
    books: dict[str, dict[str, Decimal]] = {}
    for row in rows:
        symbol = row.get("symbol")
        if symbol not in symbols or symbol in books:
            raise ValueError("unexpected or duplicate book ticker symbol")
        book = {
            "ask_price": Decimal(str(row["askPrice"])),
            "ask_quantity": Decimal(str(row["askQty"])),
            "bid_price": Decimal(str(row["bidPrice"])),
            "bid_quantity": Decimal(str(row["bidQty"])),
        }
        if min(book.values()) <= 0 or book["bid_price"] > book["ask_price"]:
            raise ValueError(f"invalid book for {symbol}")
        books[symbol] = book
    if set(books) != set(symbols):
        raise ValueError("missing frozen book ticker symbol")
    return books


def _evaluate_route(
    route: dict[str, Any],
    loan_amounts: list[Decimal],
    service_fee: Decimal,
    books: dict[str, dict[str, Decimal]],
) -> dict[str, Any]:
    pair = route["pair"]
    step = Decimal(route["lot_step_asset"])
    bonus_fraction = Decimal(route["bonus_APR_fraction"])
    bonus_days = Decimal(route["bonus_accrual_days"])
    exit_stress = Decimal(route["exit_stress_fraction"])
    evaluations: list[dict[str, Any]] = []

    for loan_amount in loan_amounts:
        disbursed = loan_amount * (Decimal(1) - service_fee)
        if pair is None:
            ask_price = bid_price = Decimal(1)
            ask_quantity = bid_quantity = Decimal("Infinity")
        else:
            book = books[pair]
            ask_price = book["ask_price"]
            ask_quantity = book["ask_quantity"]
            bid_price = book["bid_price"]
            bid_quantity = book["bid_quantity"]

        acquired_quantity = _floor_step(disbursed / ask_price, step)
        residual_usdt = disbursed - acquired_quantity * ask_price
        fixed_bonus_quantity = (
            acquired_quantity * bonus_fraction * bonus_days / Decimal(365)
        )
        exit_quantity = acquired_quantity + fixed_bonus_quantity
        current_final_usdt = exit_quantity * bid_price + residual_usdt
        stressed_exit_price = bid_price * (Decimal(1) + exit_stress)
        stressed_final_usdt = exit_quantity * stressed_exit_price + residual_usdt
        current_net_usdt = current_final_usdt - loan_amount
        stressed_net_usdt = stressed_final_usdt - loan_amount
        entry_capacity_valid = acquired_quantity <= ask_quantity
        exit_capacity_valid = exit_quantity <= bid_quantity

        evaluations.append(
            {
                "acquired_quantity": _decimal_text(acquired_quantity),
                "current_net_bips_of_loan": _decimal_text(
                    current_net_usdt / loan_amount * Decimal(10000)
                ),
                "current_net_USDT": _decimal_text(current_net_usdt),
                "disbursed_USDT": _decimal_text(disbursed),
                "entry_capacity_valid": entry_capacity_valid,
                "exit_capacity_valid": exit_capacity_valid,
                "fixed_bonus_quantity": _decimal_text(fixed_bonus_quantity),
                "loan_amount_USDT": _decimal_text(loan_amount),
                "residual_USDT": _decimal_text(residual_usdt),
                "stressed_exit_price_USDT": _decimal_text(stressed_exit_price),
                "stressed_net_bips_of_loan": _decimal_text(
                    stressed_net_usdt / loan_amount * Decimal(10000)
                ),
                "stressed_net_USDT": _decimal_text(stressed_net_usdt),
            }
        )

    candidate = all(
        row["entry_capacity_valid"]
        and row["exit_capacity_valid"]
        and Decimal(row["stressed_net_USDT"]) > 0
        for row in evaluations
    )
    return {
        "asset": route["asset"],
        "bonus_APR_fraction": route["bonus_APR_fraction"],
        "bonus_accrual_days": route["bonus_accrual_days"],
        "evaluations": evaluations,
        "fixed_bonus_only_historical_stress_candidate": candidate,
        "pair": pair,
        "promotion_end_utc": route["promotion_end_utc"],
        "real_time_APR_credited": False,
        "spot_maker_and_taker_fee_fraction": route["spot_maker_and_taker_fee_fraction"],
        "stress_fraction": route["exit_stress_fraction"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()

    contract, contract_sha256 = _read_contract(args.contract)
    request_contract = contract["market_data_request"]
    if request_contract["maximum_requests"] != 1 or not request_contract["no_retry"]:
        raise ValueError("contract must permit exactly one request with no retry")

    raw_rows, raw_evidence = _capture_once(request_contract["url"], args.raw_dir)
    frozen_at = datetime.fromisoformat(contract["frozen_at_utc"].replace("Z", "+00:00"))
    requested_before = datetime.fromisoformat(
        raw_evidence["requested_before_utc"].replace("Z", "+00:00")
    )
    if frozen_at >= requested_before:
        raise ValueError("contract freeze must precede the first request")
    loan_promotion_end = datetime.fromisoformat(
        contract["source_bound_inputs"]["lite_loan_promotion"]["end_utc"].replace(
            "Z", "+00:00"
        )
    )
    if requested_before >= loan_promotion_end:
        raise ValueError("Lite Loan promotion expired before the market request")
    for route in contract["yield_routes"]:
        promotion_end = datetime.fromisoformat(
            route["promotion_end_utc"].replace("Z", "+00:00")
        )
        if requested_before >= promotion_end:
            raise ValueError(f"{route['asset']} yield promotion already expired")

    symbols = request_contract["symbols"]
    books = _books_by_symbol(raw_rows, symbols)
    loan = contract["loan_contract"]
    evaluations = [
        _evaluate_route(
            route,
            [Decimal(value) for value in loan["loan_amounts_USDT"]],
            Decimal(loan["service_fee_fraction"]),
            books,
        )
        for route in contract["yield_routes"]
    ]
    candidates = [
        route["asset"]
        for route in evaluations
        if route["fixed_bonus_only_historical_stress_candidate"]
    ]

    result: dict[str, Any] = {
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "market_direction_forecast_required": False,
            "profitability_claim": False,
            "public_positive_fixed_bonus_historical_stress_candidates": candidates,
            "stable_edge": False,
            "status": "time_limited_public_positive_candidate_account_and_future_loss_bounds_unproved",
            "trading_authority": False,
        },
        "authority": {
            "account_requests": 0,
            "credentials_used": False,
            "funded_actions": 0,
            "orders_conversions_subscriptions_borrows_or_repays": 0,
            "public_market_data_requests": raw_evidence["request_count"],
        },
        "books": {
            symbol: {key: _decimal_text(value) for key, value in book.items()}
            for symbol, book in sorted(books.items())
        },
        "contract": {
            "path": args.contract.as_posix(),
            "sha256": contract_sha256,
        },
        "generated_at_utc": _utc_now(),
        "limitations": [
            "Historical close stress is not a bound on future depeg issuer venue custody freeze or insolvency loss.",
            "The current public zero-fee row and promotions are discretionary and do not prove exact account region or future eligibility.",
            "No account product capacity owned idle BTC collateral reward rounding redemption tax custody or operating evidence was accessed.",
            "BTC collateral remains economically material even without price-triggered liquidation during the initial term; only independently held idle BTC is in scope.",
            "The result credits no variable Real-Time APR and no random Lite Loan reward voucher.",
        ],
        "raw_evidence": raw_evidence,
        "routes": evaluations,
        "schema_version": "binance-lite-loan-stablecoin-yield-curve-v1",
    }
    result["result_sha256"] = _sha256_bytes(_canonical_bytes(result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(result) + b"\n")

    print(
        json.dumps(
            {
                "candidates": candidates,
                "request_count": raw_evidence["request_count"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
