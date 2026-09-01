"""Audit a later retained Binance Options ticker delta without network access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "binance-crypto-option-late-ticker-delta-contract-v1"
RESULT_SCHEMA_VERSION = "binance-crypto-option-late-ticker-delta-result-v1"
UNDERLYINGS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
SYMBOL_PATTERN = re.compile(
    r"^(?P<base>BTC|ETH|SOL)-(?P<expiry>[0-9]{6})-"
    r"(?P<strike>[0-9]+(?:\.[0-9]+)?)-(?P<side>C|P)$"
)


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


def _read_binding(binding: Mapping[str, object], hash_field: str) -> bytes:
    path = ROOT / str(binding["path"])
    payload = path.read_bytes()
    if _sha256(payload) != binding[hash_field]:
        raise ValueError(f"retained source hash mismatch: {path}")
    return payload


def _baseline_symbols(payload: Mapping[str, object]) -> list[str]:
    rows = _list(payload.get("optionSymbols"), name="optionSymbols")
    symbols = sorted(
        str(row["symbol"])
        for value in rows
        if isinstance(value, Mapping)
        and (row := dict(value)).get("status") == "TRADING"
        and row.get("contractType") == "CRYPTO_OPTIONS"
        and row.get("underlyingType") == "CRYPTO"
        and row.get("underlying") in UNDERLYINGS.values()
        and row.get("quoteAsset") == "USDT"
        and _decimal(row.get("unit")) == Decimal("1")
    )
    if len(symbols) != len(set(symbols)):
        raise ValueError("baseline option symbols are not unique")
    return symbols


def _ticker_rows(payload: object) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for value in _list(payload, name="option tickers"):
        row = _mapping(value, name="option ticker")
        symbol = str(row.get("symbol") or "")
        if SYMBOL_PATTERN.fullmatch(symbol) is None:
            continue
        if symbol in rows:
            raise ValueError("scoped option ticker symbols are not unique")
        rows[symbol] = row
    return rows


def _screen_rows(
    *,
    symbols: list[str],
    tickers: Mapping[str, Mapping[str, object]],
    futures_books: Mapping[str, Mapping[str, object]],
    fixed_bps: Decimal,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        match = SYMBOL_PATTERN.fullmatch(symbol)
        if match is None:
            raise ValueError(f"unexpected scoped symbol: {symbol}")
        ticker = _mapping(tickers.get(symbol), name=f"{symbol} ticker")
        base = match.group("base")
        underlying = UNDERLYINGS[base]
        future = _mapping(
            futures_books.get(underlying), name=f"{underlying} futures book"
        )
        side = "CALL" if match.group("side") == "C" else "PUT"
        symbol_strike = _decimal(match.group("strike"))
        ticker_strike = _decimal(ticker.get("strikePrice"))
        if ticker_strike != symbol_strike:
            raise ValueError(f"ticker strike disagrees with symbol: {symbol}")
        option_ask = _decimal(ticker.get("askPrice"))
        entry_field = "bidPrice" if side == "CALL" else "askPrice"
        perpetual_entry = _decimal(future.get(entry_field))
        gross = (
            perpetual_entry - ticker_strike - option_ask
            if side == "CALL"
            else ticker_strike - perpetual_entry - option_ask
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
                "option_side": side,
                "strike_USDT": format(ticker_strike, "f"),
                "option_ask_USDT": format(option_ask, "f"),
                "perpetual_entry_side": "bid" if side == "CALL" else "ask",
                "perpetual_entry_USDT": format(perpetual_entry, "f"),
                "positive_entry_sides": positive_entry,
                "gross_terminal_floor_per_underlying_unit_USDT": format(gross, "f"),
                "fixed_fee_and_basis_cost_per_underlying_unit_USDT": format(
                    fixed_cost, "f"
                ),
                "after_fixed_stress_per_underlying_unit_USDT": format(
                    after_fixed, "f"
                ),
                "after_fixed_stress_bps": format(after_fixed_bps, "f"),
                "passes_fixed_rejection_gate": positive_entry and after_fixed > 0,
                "current_exchange_info_unit_and_status_confirmed": False,
                "option_depth_quantity_verified": False,
                "accepted_edge": False,
                "deployment_ready": False,
            }
        )
    return rows


def _validate_contract(contract: Mapping[str, object], *, preflight_only: bool) -> None:
    expected_status = (
        "preflight_only_unconsumed"
        if preflight_only
        else "frozen_before_zero_network_late_ticker_delta"
    )
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected contract schema")
    if contract.get("status") != expected_status:
        raise ValueError("contract status does not match invocation mode")
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    implementation = _mapping(contract["implementation"], name="implementation")
    _read_binding(implementation, "sha256")
    retained = _mapping(contract["retained_sources"], name="retained sources")
    for key, hash_field in (
        ("baseline_exchange_info", "sha256"),
        ("baseline_request_journal", "sha256"),
        ("later_option_tickers", "sha256"),
        ("later_futures_books", "sha256"),
        ("later_request_journal", "sha256"),
        ("economic_source_contract", "sha256"),
    ):
        _read_binding(_mapping(retained[key], name=key), hash_field)
    authority = _mapping(contract["authority"], name="authority")
    if authority != {
        "account_state_accessed": False,
        "authenticated_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "new_public_requests": 0,
        "orders_quotes_transfers_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }:
        raise ValueError("unexpected authority")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    contract_bytes = args.contract.read_bytes()
    contract = _mapping(json.loads(contract_bytes), name="contract")
    _validate_contract(contract, preflight_only=args.preflight_only)
    if args.preflight_only:
        if args.output is not None:
            raise ValueError("preflight must not specify an output")
        print("preflight_passed=true")
        return 0
    if args.output is None:
        raise ValueError("frozen execution requires --output")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite retained delta evidence")

    retained = _mapping(contract["retained_sources"], name="retained sources")
    baseline = _mapping(
        json.loads(
            _read_binding(
                _mapping(retained["baseline_exchange_info"], name="baseline"),
                "sha256",
            )
        ),
        name="baseline exchangeInfo",
    )
    tickers_raw = json.loads(
        _read_binding(
            _mapping(retained["later_option_tickers"], name="tickers"), "sha256"
        )
    )
    futures_raw = _list(
        json.loads(
            _read_binding(
                _mapping(retained["later_futures_books"], name="futures"),
                "sha256",
            )
        ),
        name="futures books",
    )
    journal_lines = _read_binding(
        _mapping(retained["later_request_journal"], name="journal"), "sha256"
    ).decode("ascii").splitlines()
    journal = [_mapping(json.loads(line), name="journal row") for line in journal_lines]
    if len(journal) != 2:
        raise ValueError("later request journal must contain exactly two rows")
    option_receipt, futures_receipt = journal
    if (
        option_receipt.get("url") != "https://eapi.binance.com/eapi/v1/ticker"
        or futures_receipt.get("url")
        != "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
        or option_receipt.get("status_code") != 200
        or futures_receipt.get("status_code") != 200
    ):
        raise ValueError("unexpected later price receipt")
    ticker_binding = _mapping(retained["later_option_tickers"], name="tickers")
    futures_binding = _mapping(retained["later_futures_books"], name="futures")
    if (
        option_receipt.get("response_sha256") != ticker_binding["sha256"]
        or futures_receipt.get("response_sha256") != futures_binding["sha256"]
    ):
        raise ValueError("later journal does not bind retained price bytes")

    baseline_symbols = _baseline_symbols(baseline)
    tickers = _ticker_rows(tickers_raw)
    new_symbols = sorted(set(tickers) - set(baseline_symbols))
    expected_new = [
        str(value)
        for value in _list(contract["expected_new_symbols"], name="expected new symbols")
    ]
    if new_symbols != expected_new:
        raise ValueError("retained late ticker delta differs from the frozen population")
    futures_books = {
        str(row["symbol"]): row
        for row in (_mapping(value, name="futures book") for value in futures_raw)
    }
    rows = _screen_rows(
        symbols=new_symbols,
        tickers=tickers,
        futures_books=futures_books,
        fixed_bps=_decimal(contract["fixed_fee_and_basis_stress_bps"]),
    )
    survivors = [row for row in rows if row["passes_fixed_rejection_gate"]]
    survivors.sort(
        key=lambda row: (-_decimal(row["after_fixed_stress_bps"]), str(row["symbol"]))
    )
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": args.contract.as_posix(),
            "sha256": _sha256(contract_bytes),
            "canonical_sha256": contract["contract_sha256"],
        },
        "implementation": contract["implementation"],
        "authority": contract["authority"],
        "retained_sources": retained,
        "population": {
            "baseline_eligible_symbol_count": len(baseline_symbols),
            "later_scoped_ticker_symbol_count": len(tickers),
            "new_symbol_count": len(new_symbols),
            "new_symbols": new_symbols,
            "positive_entry_side_count": sum(
                bool(row["positive_entry_sides"]) for row in rows
            ),
            "gross_positive_count": sum(
                bool(row["positive_entry_sides"])
                and _decimal(row["gross_terminal_floor_per_underlying_unit_USDT"])
                > 0
                for row in rows
            ),
            "after_fixed_stress_positive_count": len(survivors),
        },
        "all_rows": rows,
        "fixed_stress_survivors": survivors,
        "adjudication": {
            "literal_rank_47_new_population_trigger_satisfied": True,
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "option_depth_requests": 0,
            "new_public_requests": 0,
            "next_action": (
                "freeze_one_current_exchange_info_and_depth_confirmation_contract_for_only_the_exact_survivors"
                if survivors
                else "terminalize_the_exact_late_ticker_delta_and_stop_without_exchange_info_depth_funding_or_account_requests"
            ),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["population"], indent=2))
    print(json.dumps(result["fixed_stress_survivors"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
