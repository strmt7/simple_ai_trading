"""Adjudicate the frozen distinct Binance option/perpetual price prefilter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "binance-crypto-option-distinct-price-prefilter-contract-v2"
RESULT_SCHEMA = "binance-crypto-option-distinct-price-prefilter-result-v2"
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


def _canonical_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _verify_self_hash(value: Mapping[str, object], field: str, name: str) -> None:
    if _canonical_hash(value, field) != value.get(field):
        raise ValueError(f"{name} canonical hash mismatch")


def _decimal(value: object) -> Decimal:
    text = str(value or "0")
    return Decimal(text)


def _load_source_result(binding: Mapping[str, object]) -> tuple[dict[str, Any], bytes]:
    result_path = ROOT / str(binding["result_path"])
    source = _load_object(result_path)
    _verify_self_hash(source, "result_sha256", result_path.name)
    if source.get("contract") != {
        "path": binding.get("contract_path"),
        "sha256": binding.get("contract_sha256"),
    }:
        raise ValueError(f"source contract binding mismatch: {result_path.name}")
    if source.get("source_gate", {}).get("passed") is not True:
        raise ValueError(f"source gate failed: {result_path.name}")
    receipt = source.get("capture", {}).get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"source receipt missing: {result_path.name}")
    raw_path = ROOT / str(receipt["raw_path"])
    raw = raw_path.read_bytes()
    if _sha256(raw) != receipt.get("response_sha256"):
        raise ValueError(f"source raw hash mismatch: {raw_path.name}")
    return source, raw


def _ticker_map(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("option ticker response must be a list")
    rows: dict[str, dict[str, Any]] = {}
    for candidate in value:
        if not isinstance(candidate, Mapping):
            continue
        row = dict(candidate)
        symbol = str(row.get("symbol") or "")
        if SYMBOL_PATTERN.fullmatch(symbol) is None:
            continue
        if symbol in rows:
            raise ValueError("option ticker symbols are not unique")
        rows[symbol] = row
    return rows


def _futures_map(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("futures book response must be a list")
    rows: dict[str, dict[str, Any]] = {}
    for candidate in value:
        if not isinstance(candidate, Mapping):
            continue
        row = dict(candidate)
        symbol = str(row.get("symbol") or "")
        if symbol not in UNDERLYINGS.values():
            continue
        if symbol in rows:
            raise ValueError("scoped futures books are not unique")
        rows[symbol] = row
    return rows


def _metadata_map(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not isinstance(value.get("optionSymbols"), list):
        raise ValueError("current exchangeInfo is invalid")
    rows: dict[str, dict[str, Any]] = {}
    for candidate in value["optionSymbols"]:
        if not isinstance(candidate, Mapping):
            continue
        row = dict(candidate)
        symbol = str(row.get("symbol") or "")
        if SYMBOL_PATTERN.fullmatch(symbol) is None:
            continue
        if symbol in rows:
            raise ValueError("exchangeInfo symbols are not unique")
        rows[symbol] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite price-prefilter evidence")

    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unexpected contract schema")
    _verify_self_hash(contract, "contract_sha256", "price prefilter contract")
    implementation = contract.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("implementation binding is missing")
    if _sha256(Path(__file__).read_bytes()) != implementation.get("sha256"):
        raise ValueError("implementation hash mismatch")

    population_binding = contract.get("population_result")
    if not isinstance(population_binding, Mapping):
        raise ValueError("population result binding is missing")
    population = _load_object(ROOT / str(population_binding["path"]))
    _verify_self_hash(population, "result_sha256", "population result")
    if population.get("result_sha256") != population_binding.get("result_sha256"):
        raise ValueError("population result hash mismatch")
    symbols = population["population"]["distinct_unscreened_symbols"]
    if not isinstance(symbols, list) or len(symbols) != 354:
        raise ValueError("unexpected distinct population size")
    if symbols != sorted(set(symbols)):
        raise ValueError("distinct population is not sorted and unique")
    symbols_hash = _sha256(("\n".join(symbols) + "\n").encode("ascii"))
    if symbols_hash != contract.get("population_rule", {}).get(
        "distinct_unscreened_symbols_sha256"
    ):
        raise ValueError("distinct population hash mismatch")

    sources = contract.get("source_results")
    if not isinstance(sources, Mapping):
        raise ValueError("source result bindings are missing")
    option_source, option_raw = _load_source_result(sources["option_tickers"])
    futures_source, futures_raw = _load_source_result(sources["futures_books"])
    tickers = _ticker_map(json.loads(option_raw))
    futures = _futures_map(json.loads(futures_raw))

    current_path = ROOT / population["source_binding"]["raw_path"]
    current_raw = current_path.read_bytes()
    if _sha256(current_raw) != population["source_binding"]["raw_sha256"]:
        raise ValueError("population exchangeInfo source hash mismatch")
    metadata = _metadata_map(json.loads(current_raw))
    fixed_bps = _decimal(contract["economic_gate"]["fixed_stress_bps"])
    if fixed_bps != Decimal("33.5"):
        raise ValueError("unexpected fixed stress")

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        match = SYMBOL_PATTERN.fullmatch(symbol)
        if match is None:
            raise ValueError(f"invalid scoped symbol: {symbol}")
        meta = metadata.get(symbol)
        if meta is None:
            raise ValueError(f"current metadata missing: {symbol}")
        underlying = UNDERLYINGS[match.group("base")]
        unit = _decimal(meta.get("unit"))
        metadata_confirmed = (
            meta.get("underlying") == underlying
            and meta.get("status") == "TRADING"
            and meta.get("contractType") == "CRYPTO_OPTIONS"
            and meta.get("underlyingType") == "CRYPTO"
            and meta.get("quoteAsset") == "USDT"
            and unit > 0
        )
        if not metadata_confirmed:
            raise ValueError(f"current metadata gate failed: {symbol}")
        ticker = tickers.get(symbol)
        if ticker is None:
            raise ValueError(f"current ticker missing: {symbol}")
        future = futures.get(underlying)
        if future is None:
            raise ValueError(f"current futures book missing: {underlying}")

        strike = _decimal(ticker.get("strikePrice"))
        symbol_strike = _decimal(match.group("strike"))
        if strike != symbol_strike:
            raise ValueError(f"ticker strike mismatch: {symbol}")
        ask = _decimal(ticker.get("askPrice"))
        is_call = match.group("side") == "C"
        entry_side = "bidPrice" if is_call else "askPrice"
        perpetual_entry = _decimal(future.get(entry_side))
        gross = (
            perpetual_entry - strike - ask
            if is_call
            else strike - perpetual_entry - ask
        )
        fixed_cost = perpetual_entry * fixed_bps / Decimal("10000")
        after_fixed = gross - fixed_cost
        positive_entry = ask > 0 and perpetual_entry > 0
        passes = positive_entry and after_fixed > 0
        after_fixed_bps = (
            after_fixed / perpetual_entry * Decimal("10000")
            if perpetual_entry > 0
            else Decimal("-Infinity")
        )
        rows.append(
            {
                "symbol": symbol,
                "underlying": underlying,
                "option_side": "CALL" if is_call else "PUT",
                "expiry_date_ms": meta.get("expiryDate"),
                "unit": format(unit, "f"),
                "strike_USDT": format(strike, "f"),
                "option_ask_USDT": format(ask, "f"),
                "perpetual_entry_side": "bid" if is_call else "ask",
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
                "passes_fixed_rejection_gate": passes,
                "current_exchange_info_unit_and_status_confirmed": True,
                "option_depth_quantity_verified": False,
                "accepted_edge": False,
                "deployment_ready": False,
            }
        )

    survivors = [row for row in rows if row["passes_fixed_rejection_gate"]]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": contract["contract_sha256"],
        },
        "population_result": population_binding,
        "capture": {
            "option_ticker_source_result_sha256": option_source["result_sha256"],
            "futures_book_source_result_sha256": futures_source["result_sha256"],
            "capture_skew_ms": abs(
                option_source["capture"]["receipt"]["requested_at_ms"]
                - futures_source["capture"]["receipt"]["requested_at_ms"]
            ),
        },
        "population": {
            "distinct_unscreened_symbol_count": len(symbols),
            "ticker_present_count": sum(symbol in tickers for symbol in symbols),
            "positive_entry_side_count": sum(
                bool(row["positive_entry_sides"]) for row in rows
            ),
            "gross_positive_count": sum(
                _decimal(row["gross_terminal_floor_per_underlying_unit_USDT"]) > 0
                and bool(row["positive_entry_sides"])
                for row in rows
            ),
            "after_fixed_stress_positive_count": len(survivors),
        },
        "fixed_stress_survivors": survivors,
        "all_rows": rows,
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "option_depth_requests": 0,
            "funding_requests": 0,
            "next_action": (
                "freeze_a_separate_full_cost_tick_funding_and_quantity_stress_before_any_depth_access"
                if survivors
                else "terminalize_the_exact_354_symbol_population_and_stop_without_depth_funding_fee_account_credential_order_or_fund_access"
            ),
        },
        "authority": {
            "account_state_accessed": False,
            "authenticated_requests": 0,
            "credentials_used": False,
            "funds_used": False,
            "public_unauthenticated_GET_requests": 2,
            "orders_quotes_transfers_or_wallet_actions": 0,
            "paper_or_live_trading_authority": False,
            "protected_capture_touched": False,
        },
        "implementation": implementation,
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    write_bytes_atomic(
        args.output, (_canonical_json(result) + "\n").encode("ascii")
    )
    print(
        _canonical_json(
            {
                "screened": len(rows),
                "positive_entries": result["population"]["positive_entry_side_count"],
                "gross_positive": result["population"]["gross_positive_count"],
                "after_fixed_stress_positive": len(survivors),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
