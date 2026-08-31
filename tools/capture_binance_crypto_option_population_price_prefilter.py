"""Capture one frozen public price prefilter for a new crypto-option population."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "binance-crypto-option-population-price-prefilter-contract-v1"
RESULT_SCHEMA_VERSION = "binance-crypto-option-population-price-prefilter-result-v1"
OPTIONS_TICKER_URL = "https://eapi.binance.com/eapi/v1/ticker"
FUTURES_BOOK_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _decimal(value: object) -> Decimal:
    text = str(value)
    return Decimal(text) if text else Decimal(0)


def _validate_binding(binding: Mapping[str, object], hash_field: str) -> bytes:
    path = ROOT / str(binding["path"])
    payload = path.read_bytes()
    if _sha256(payload) != binding[hash_field]:
        raise ValueError(f"retained source hash mismatch: {path}")
    return payload


def _validate_contract(contract: Mapping[str, object], *, preflight_only: bool) -> None:
    expected_status = (
        "preflight_only_unconsumed"
        if preflight_only
        else "frozen_before_two_public_price_requests"
    )
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected contract schema")
    if contract.get("status") != expected_status:
        raise ValueError("contract status does not match invocation mode")
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    implementation = _mapping(contract["implementation"], name="implementation")
    _validate_binding(implementation, "sha256")
    retained = _mapping(contract["retained_sources"], name="retained sources")
    _validate_binding(
        _mapping(retained["current_exchange_info"], name="exchange"), "sha256"
    )
    delta_bytes = _validate_binding(
        _mapping(retained["population_delta"], name="population delta"),
        "file_sha256",
    )
    delta = _mapping(json.loads(delta_bytes), name="population delta")
    delta_binding = _mapping(retained["population_delta"], name="population delta")
    if delta.get("result_sha256") != delta_binding["canonical_result_sha256"]:
        raise ValueError("population delta canonical hash mismatch")
    if _canonical_hash(delta, "result_sha256") != delta.get("result_sha256"):
        raise ValueError("population delta embedded hash mismatch")
    _validate_binding(
        _mapping(
            retained["endpoint_and_economic_source_contract"],
            name="endpoint source contract",
        ),
        "file_sha256",
    )
    requests_contract = _list(contract["requests"], name="requests")
    expected = [
        {"method": "GET", "name": "options-all-tickers", "url": OPTIONS_TICKER_URL},
        {
            "method": "GET",
            "name": "futures-all-book-tickers",
            "url": FUTURES_BOOK_URL,
        },
    ]
    if requests_contract != expected:
        raise ValueError("unexpected request plan")
    authority = _mapping(contract["authority"], name="authority")
    if authority != {
        "account_state_accessed": False,
        "authenticated_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "maximum_public_unauthenticated_GET_requests": 2,
        "orders_quotes_transfers_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }:
        raise ValueError("unexpected authority")


def _capture(
    session: requests.Session,
    *,
    url: str,
    name: str,
    raw_dir: Path,
    journal: Path,
) -> tuple[object, dict[str, object]]:
    started_ms = time.time_ns() // 1_000_000
    response = session.get(url, timeout=30)
    completed_ms = time.time_ns() // 1_000_000
    payload = response.content
    raw_path = raw_dir / f"{name}.raw"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite retained response: {raw_path}")
    write_bytes_atomic(raw_path, payload)
    receipt = {
        "name": name,
        "method": "GET",
        "url": response.url,
        "status_code": response.status_code,
        "requested_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "response_bytes": len(payload),
        "response_sha256": _sha256(payload),
        "raw_path": raw_path.as_posix(),
    }
    with journal.open("ab") as stream:
        stream.write((_canonical_json(receipt) + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    response.raise_for_status()
    try:
        return response.json(), receipt
    except requests.JSONDecodeError as exc:
        raise ValueError(f"{name} did not return JSON") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    contract_bytes = args.contract.read_bytes()
    contract = _mapping(json.loads(contract_bytes), name="contract")
    _validate_contract(contract, preflight_only=args.preflight_only)
    if args.preflight_only:
        if any(
            value is not None for value in (args.raw_dir, args.journal, args.output)
        ):
            raise ValueError("preflight must not specify capture outputs")
        print("preflight_passed=true")
        return 0
    if args.raw_dir is None or args.journal is None or args.output is None:
        raise ValueError("frozen execution requires raw-dir, journal, and output")
    if args.journal.exists() or args.output.exists():
        raise FileExistsError("refusing to overwrite retained price evidence")
    if args.raw_dir.exists() and any(args.raw_dir.iterdir()):
        raise FileExistsError("refusing to reuse non-empty raw directory")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        }
    )
    option_tickers_raw, option_receipt = _capture(
        session,
        url=OPTIONS_TICKER_URL,
        name="options-all-tickers",
        raw_dir=args.raw_dir,
        journal=args.journal,
    )
    futures_books_raw, futures_receipt = _capture(
        session,
        url=FUTURES_BOOK_URL,
        name="futures-all-book-tickers",
        raw_dir=args.raw_dir,
        journal=args.journal,
    )
    capture_skew_ms = int(futures_receipt["requested_at_ms"]) - int(
        option_receipt["completed_at_ms"]
    )
    if capture_skew_ms > int(contract["maximum_capture_skew_ms"]):
        raise ValueError("price capture exceeded the frozen synchronization window")

    retained = _mapping(contract["retained_sources"], name="retained sources")
    delta_binding = _mapping(retained["population_delta"], name="population delta")
    delta = _mapping(
        json.loads(_validate_binding(delta_binding, "file_sha256")), name="delta"
    )
    population = _mapping(delta["population"], name="delta population")
    new_symbols = [
        str(value) for value in _list(population["new_symbols"], name="new symbols")
    ]
    if len(new_symbols) != int(population["new_symbol_count"]) or not new_symbols:
        raise ValueError(
            "population delta does not expose a non-empty exact new symbol set"
        )

    exchange_binding = _mapping(retained["current_exchange_info"], name="exchange")
    exchange = _mapping(
        json.loads(_validate_binding(exchange_binding, "sha256")), name="exchangeInfo"
    )
    option_rows = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="option symbol")
            for value in _list(exchange.get("optionSymbols"), name="option symbols")
        )
    }
    option_tickers = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="option ticker")
            for value in _list(option_tickers_raw, name="option tickers")
        )
    }
    futures_books = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="futures book")
            for value in _list(futures_books_raw, name="futures books")
        )
    }
    fixed_bps = _decimal(contract["fixed_fee_and_basis_stress_bps"])
    rows: list[dict[str, object]] = []
    for symbol in new_symbols:
        option = _mapping(option_rows.get(symbol), name=f"{symbol} option")
        ticker = _mapping(option_tickers.get(symbol), name=f"{symbol} ticker")
        underlying = str(option["underlying"])
        future = _mapping(futures_books.get(underlying), name=f"{underlying} book")
        side = str(option["side"])
        if side not in {"CALL", "PUT"}:
            raise ValueError(f"unsupported option side for {symbol}")
        option_ask = _decimal(ticker.get("askPrice"))
        perpetual_entry = _decimal(
            future.get("bidPrice" if side == "CALL" else "askPrice")
        )
        strike = _decimal(option["strikePrice"])
        gross = (
            perpetual_entry - strike - option_ask
            if side == "CALL"
            else strike - perpetual_entry - option_ask
        )
        positive_entry = option_ask > 0 and perpetual_entry > 0
        fixed_cost = perpetual_entry * fixed_bps / Decimal("10000")
        after_fixed = gross - fixed_cost
        after_fixed_bps = (
            after_fixed / perpetual_entry * Decimal("10000")
            if perpetual_entry > 0
            else Decimal("-Infinity")
        )
        rows.append(
            {
                "symbol": symbol,
                "underlying": underlying,
                "expiry_date_ms": int(option["expiryDate"]),
                "option_side": side,
                "strike_USDT": format(strike, "f"),
                "option_ask_USDT": format(option_ask, "f"),
                "perpetual_entry_side": "bid" if side == "CALL" else "ask",
                "perpetual_entry_USDT": format(perpetual_entry, "f"),
                "positive_entry_sides": positive_entry,
                "gross_terminal_floor_per_unit_USDT": format(gross, "f"),
                "fixed_fee_and_basis_cost_per_unit_USDT": format(fixed_cost, "f"),
                "after_fixed_stress_per_unit_USDT": format(after_fixed, "f"),
                "after_fixed_stress_bps": format(after_fixed_bps, "f"),
                "passes_fixed_rejection_gate": positive_entry and after_fixed > 0,
                "option_depth_quantity_verified": False,
                "accepted_edge": False,
                "deployment_ready": False,
            }
        )
    survivors = [row for row in rows if row["passes_fixed_rejection_gate"]]
    survivors.sort(
        key=lambda row: (-_decimal(row["after_fixed_stress_bps"]), str(row["symbol"]))
    )
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "path": args.contract.as_posix(),
            "sha256": _sha256(contract_bytes),
            "canonical_sha256": contract["contract_sha256"],
        },
        "implementation": contract["implementation"],
        "authority": {
            "public_unauthenticated_GET_requests": 2,
            "authenticated_requests": 0,
            "credentials_used": False,
            "account_state_accessed": False,
            "orders_quotes_transfers_or_wallet_actions": 0,
            "funds_used": False,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "journal_path": args.journal.as_posix(),
            "capture_skew_ms": capture_skew_ms,
            "sources": [option_receipt, futures_receipt],
        },
        "population": {
            "new_symbol_count": len(new_symbols),
            "positive_entry_side_count": sum(
                bool(row["positive_entry_sides"]) for row in rows
            ),
            "gross_positive_count": sum(
                bool(row["positive_entry_sides"])
                and _decimal(row["gross_terminal_floor_per_unit_USDT"]) > 0
                for row in rows
            ),
            "after_fixed_stress_positive_count": len(survivors),
        },
        "all_rows": rows,
        "fixed_stress_survivors": survivors,
        "adjudication": {
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "option_depth_requests": 0,
            "next_action": (
                "freeze_one_zero_network_complete_retained_stress_for_the_exact_survivors_before_any_depth_request"
                if survivors
                else "terminalize_the_exact_new_population_and_stop_without_depth_funding_or_account_requests"
            ),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["population"], indent=2))
    print(json.dumps(result["fixed_stress_survivors"][:10], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
