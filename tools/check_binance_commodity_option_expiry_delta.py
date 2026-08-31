"""Run one frozen public Binance commodity-option expiry inventory delta."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


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


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _selected_rows(
    payload: Mapping[str, object],
    *,
    contract_type: str = "TRADFI_OPTIONS",
    underlying_type: str = "COMMODITY",
    underlyings: set[str] | None = None,
) -> list[dict[str, object]]:
    rows = payload.get("optionSymbols")
    if not isinstance(rows, list):
        raise ValueError("exchangeInfo optionSymbols must be a list")
    selected = [
        _mapping(value, name="option symbol")
        for value in rows
        if isinstance(value, Mapping)
        and value.get("status") == "TRADING"
        and value.get("contractType") == contract_type
        and value.get("underlyingType") == underlying_type
        and (underlyings is None or value.get("underlying") in underlyings)
    ]
    selected.sort(key=lambda row: str(row["symbol"]))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.journal.exists() or args.output.exists():
        raise FileExistsError("refusing to overwrite retained delta evidence")
    if args.raw_dir.exists() and any(args.raw_dir.iterdir()):
        raise FileExistsError("refusing to reuse a non-empty raw directory")

    contract_bytes = args.contract.read_bytes()
    contract = _mapping(json.loads(contract_bytes), name="contract")
    schema_version = str(contract.get("schema_version") or "")
    supported_schemas = {
        "binance-commodity-option-expiry-delta-contract-v1",
        "binance-stock-option-inventory-contract-v1",
    }
    if (
        schema_version not in supported_schemas
        or contract.get("status") != "frozen_before_one_public_inventory_request"
    ):
        raise ValueError("unexpected or unfrozen contract")
    endpoint = str(_mapping(contract["capture"], name="capture")["endpoint"])

    if schema_version == "binance-stock-option-inventory-contract-v1":
        population_filter = _mapping(
            contract["population_filter"], name="population filter"
        )
        if population_filter != {
            "contract_type": "TRADFI_OPTIONS",
            "status": "TRADING",
            "underlying_type": "EQUITY",
        }:
            raise ValueError("unexpected stock-option population filter")

    started_ms = time.time_ns() // 1_000_000
    frozen_text = str(contract.get("frozen_at_utc") or "")
    try:
        frozen_at = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("contract frozen_at_utc must be ISO 8601") from exc
    if frozen_at.tzinfo is None:
        raise ValueError("contract frozen_at_utc must include an offset")
    frozen_ms = int(frozen_at.astimezone(timezone.utc).timestamp() * 1000)
    if frozen_ms > started_ms:
        raise ValueError("contract frozen_at_utc is later than request start")
    if endpoint != "https://eapi.binance.com/eapi/v1/exchangeInfo":
        raise ValueError("inventory endpoint is not the exact public exchangeInfo route")
    response = requests.get(
        endpoint,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        },
        timeout=30,
    )
    completed_ms = time.time_ns() // 1_000_000
    payload = response.content
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / "options-exchange-info.raw"
    write_bytes_atomic(raw_path, payload)
    receipt = {
        "name": "options-exchange-info",
        "method": "GET",
        "url": response.url,
        "status_code": response.status_code,
        "requested_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "response_bytes": len(payload),
        "response_sha256": _sha256(payload),
    }
    write_bytes_atomic(
        args.journal,
        (_canonical_json(receipt) + "\n").encode("ascii"),
    )
    response.raise_for_status()

    exchange_info = _mapping(response.json(), name="exchangeInfo")
    if schema_version == "binance-stock-option-inventory-contract-v1":
        selected = _selected_rows(
            exchange_info,
            contract_type="TRADFI_OPTIONS",
            underlying_type="EQUITY",
        )
    else:
        selected = _selected_rows(
            exchange_info,
            underlyings={"XAGUSDT", "XAUUSDT"},
        )
    symbols = [str(row["symbol"]) for row in selected]
    groups: dict[tuple[str, int], int] = {}
    for row in selected:
        key = (str(row["underlying"]), int(row["expiryDate"]))
        groups[key] = groups.get(key, 0) + 1
    expiry_groups = [
        {
            "underlying": underlying,
            "expiry_date_ms": expiry,
            "symbol_count": count,
        }
        for (underlying, expiry), count in sorted(groups.items())
    ]
    new_expiries: list[int] = []
    removed_expiries: list[int] = []
    if schema_version == "binance-commodity-option-expiry-delta-contract-v1":
        baseline = _mapping(contract["retained_baseline"], name="retained baseline")
        baseline_groups = baseline["expiry_groups"]
        if not isinstance(baseline_groups, list):
            raise ValueError("baseline expiry_groups must be a list")
        baseline_expiries = {
            int(_mapping(row, name="baseline expiry group")["expiry_date_ms"])
            for row in baseline_groups
        }
        current_expiries = {expiry for _, expiry in groups}
        new_expiries = sorted(current_expiries - baseline_expiries)
        removed_expiries = sorted(baseline_expiries - current_expiries)

    is_stock_inventory = schema_version == "binance-stock-option-inventory-contract-v1"
    population: dict[str, object]
    adjudication: dict[str, object]
    if is_stock_inventory:
        population = {
            "active_stock_option_count": len(selected),
            "sorted_symbol_population_sha256": _sha256(
                "\n".join(symbols).encode("ascii")
            ),
            "underlyings": sorted({str(row["underlying"]) for row in selected}),
            "expiry_groups": expiry_groups,
        }
        adjudication = {
            "stock_option_inventory_trigger_satisfied": bool(selected),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "next_action": (
                "freeze_one_separate_exact_stock_option_perpetual_economics_contract_before_any_ticker_futures_or_funding_request"
                if selected
                else "stop_without_tickers_futures_books_premium_or_funding_and_wait_for_an_official_stock_option_listing"
            ),
        }
    else:
        population = {
            "active_commodity_option_count": len(selected),
            "sorted_symbol_population_sha256": _sha256(
                "\n".join(symbols).encode("ascii")
            ),
            "expiry_groups": expiry_groups,
            "new_expiry_date_ms": new_expiries,
            "removed_expiry_date_ms": removed_expiries,
        }
        adjudication = {
            "new_listed_expiry_trigger_satisfied": bool(new_expiries),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "next_action": (
                "freeze_one_separate_exact_population_economics_contract_before_any_ticker_or_futures_request"
                if new_expiries
                else "stop_without_tickers_books_premium_or_funding_and_do_not_poll_unchanged_inventory"
            ),
        }
    result: dict[str, object] = {
        "schema_version": (
            "binance-stock-option-inventory-result-v1"
            if is_stock_inventory
            else "binance-commodity-option-expiry-delta-result-v1"
        ),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "path": args.contract.as_posix(),
            "sha256": _sha256(contract_bytes),
        },
        "authority": {
            "public_unauthenticated_GET_requests": 1,
            "authenticated_requests": 0,
            "account_state_accessed": False,
            "orders_quotes_transfers_or_wallet_actions": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "journal_path": args.journal.as_posix(),
            "source": receipt,
        },
        "population": population,
        "adjudication": adjudication,
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["population"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
