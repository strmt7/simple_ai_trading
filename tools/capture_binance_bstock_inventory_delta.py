"""Capture one frozen public bStock inventory delta and conditional futures metadata."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.request


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _request(item: dict[str, str], raw_dir: Path) -> dict[str, Any]:
    requested_before_ms = int(time.time() * 1000)
    request = urllib.request.Request(
        item["url"],
        method="GET",
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-research/1",
        },
    )
    status_code = 0
    final_url = item["url"]
    content_type = ""
    payload = b""
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status_code = int(response.status)
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            payload = response.read()
    except Exception as exc:  # the one-use outcome is retained and never retried
        error = f"{type(exc).__name__}:{exc}"
    received_after_ms = int(time.time() * 1000)
    raw_path = raw_dir / f"{item['label']}.json"
    _write_atomic(raw_path, payload)
    return {
        "content_type": content_type,
        "elapsed_ms": received_after_ms - requested_before_ms,
        "error": error,
        "final_url": final_url,
        "label": item["label"],
        "method": "GET",
        "payload_bytes": len(payload),
        "payload_sha256": _sha256(payload),
        "raw_path": raw_path.as_posix(),
        "received_after_ms": received_after_ms,
        "requested_before_ms": requested_before_ms,
        "status_code": status_code,
        "url": item["url"],
    }


def _load_success(receipt: dict[str, Any]) -> Any:
    if receipt["status_code"] != 200 or receipt["error"]:
        raise ValueError("request_not_successful")
    return json.loads(Path(receipt["raw_path"]).read_bytes())


def _inventory_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("code") != "000000":
        raise ValueError("inventory_envelope_invalid")
    raw_rows = payload.get("data")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("inventory_rows_missing")
    required = {
        "ticker",
        "type",
        "symbol",
        "asset",
        "cs",
        "multiplier",
        "contractAddress",
        "chainId",
        "d",
    }
    rows: list[dict[str, Any]] = []
    tickers: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict) or not required.issubset(raw_row):
            raise ValueError("inventory_row_identity_invalid")
        if raw_row["type"] != 3:
            raise ValueError("inventory_row_type_invalid")
        ticker = str(raw_row["ticker"])
        if ticker in tickers:
            raise ValueError("inventory_ticker_duplicate")
        tickers.add(ticker)
        rows.append(raw_row)
    return rows


def _is_exact_multiplier(row: dict[str, Any]) -> bool:
    try:
        return Decimal(str(row["multiplier"])) == Decimal(1)
    except InvalidOperation:
        return False


def _future_matches(payload: Any, new_exact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("futures_exchange_info_invalid")
    candidates = {f"{row['ticker']}USDT": row for row in new_exact_rows}
    matches: list[dict[str, Any]] = []
    for symbol in payload["symbols"]:
        if not isinstance(symbol, dict) or symbol.get("symbol") not in candidates:
            continue
        if (
            symbol.get("contractType") == "TRADIFI_PERPETUAL"
            and symbol.get("status") == "TRADING"
            and symbol.get("underlyingType") == "EQUITY"
        ):
            source = candidates[str(symbol["symbol"])]
            matches.append(
                {
                    "bstock_multiplier": str(source["multiplier"]),
                    "bstock_spot_symbol": str(source["cs"]),
                    "futures_contract_type": str(symbol["contractType"]),
                    "futures_status": str(symbol["status"]),
                    "futures_symbol": str(symbol["symbol"]),
                    "futures_underlying_type": str(symbol["underlyingType"]),
                    "ticker": str(source["ticker"]),
                }
            )
    return sorted(matches, key=lambda row: row["ticker"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    claimed_contract_hash = str(contract.pop("contract_sha256"))
    actual_contract_hash = _sha256(_canonical(contract))
    if claimed_contract_hash != actual_contract_hash:
        raise SystemExit(
            f"contract hash mismatch: {claimed_contract_hash} != {actual_contract_hash}"
        )
    contract["contract_sha256"] = claimed_contract_hash

    baseline_path = Path(contract["baseline"]["path"])
    baseline_payload = baseline_path.read_bytes()
    if _sha256(baseline_payload) != contract["baseline"]["response_sha256"]:
        raise SystemExit("baseline hash mismatch")
    baseline_rows = _inventory_rows(json.loads(baseline_payload))
    if len(baseline_rows) != contract["baseline"]["row_count"]:
        raise SystemExit("baseline row count mismatch")

    inventory_receipt = _request(contract["inventory_request"], args.raw_dir)
    receipts = [inventory_receipt]
    status = "inventory_request_or_schema_failed"
    current_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    removed_tickers: list[str] = []
    new_exact_rows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    try:
        current_rows = _inventory_rows(_load_success(inventory_receipt))
        baseline_by_ticker = {str(row["ticker"]): row for row in baseline_rows}
        current_by_ticker = {str(row["ticker"]): row for row in current_rows}
        new_rows = [
            current_by_ticker[ticker]
            for ticker in sorted(current_by_ticker.keys() - baseline_by_ticker.keys())
        ]
        removed_tickers = sorted(baseline_by_ticker.keys() - current_by_ticker.keys())
        new_exact_rows = [row for row in new_rows if _is_exact_multiplier(row)]
        if new_exact_rows:
            futures_receipt = _request(contract["conditional_request"], args.raw_dir)
            receipts.append(futures_receipt)
            matches = _future_matches(_load_success(futures_receipt), new_exact_rows)
        if matches:
            status = "new_exact_multiplier_matching_tradifi_perpetual_prefilter_trigger_satisfied"
        elif new_rows:
            status = "inventory_changed_without_new_exact_matching_tradifi_perpetual"
        else:
            status = "no_new_bstock_rows_inventory_trigger_not_satisfied"
    except (ValueError, json.JSONDecodeError):
        pass

    journal = {
        "conditional_request_executed": len(receipts) == 2,
        "contract_sha256": claimed_contract_hash,
        "request_count": len(receipts),
        "receipts": receipts,
        "retry_count": 0,
        "schema_version": "binance-bstock-inventory-delta-journal-v1",
        "status": status,
    }
    journal["journal_sha256"] = _sha256(_canonical(journal))
    _write_atomic(args.journal, _canonical(journal))

    public_trigger_satisfied = bool(matches)
    result = {
        "accepted_edge": False,
        "baseline_row_count": len(baseline_rows),
        "conditional_request_executed": len(receipts) == 2,
        "contract_sha256": claimed_contract_hash,
        "current_row_count": len(current_rows),
        "deployment_ready": False,
        "journal_path": args.journal.as_posix(),
        "journal_sha256": journal["journal_sha256"],
        "market_direction_forecast_required": False,
        "matching_unscreened_pairs": matches,
        "new_exact_multiplier_tickers": sorted(
            str(row["ticker"]) for row in new_exact_rows
        ),
        "new_tickers": sorted(str(row["ticker"]) for row in new_rows),
        "next_selected_ticker": matches[0]["ticker"] if matches else None,
        "profitability_claim": False,
        "public_prefilter_trigger_satisfied": public_trigger_satisfied,
        "removed_tickers": removed_tickers,
        "request_count": len(receipts),
        "schema_version": "binance-bstock-inventory-delta-result-v1",
        "status": status,
        "trading_authority": False,
    }
    result["result_sha256"] = _sha256(_canonical(result))
    _write_atomic(args.result, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
