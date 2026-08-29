"""Execute the frozen one-use public GLW dividend-funding trigger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any


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
        item["url"], method="GET", headers={"User-Agent": "simple-ai-trading-public-research/1"}
    )
    status_code = 0
    final_url = item["url"]
    content_type = ""
    payload = b""
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status_code = int(response.status)
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            payload = response.read()
    except Exception as exc:  # one-use failure is evidence; never retry
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


def _load_json(receipt: dict[str, Any]) -> Any:
    if receipt["status_code"] != 200 or receipt["error"]:
        raise ValueError("request_not_successful")
    return json.loads(Path(receipt["raw_path"]).read_bytes())


def _history_gate(contract: dict[str, Any], rows: Any) -> tuple[list[dict[str, str]], str]:
    if not isinstance(rows, list):
        return [], "history_payload_not_list"
    dividend = Decimal(contract["acceptance"]["declared_gross_dividend_per_share_usd"])
    tolerance = Decimal(contract["acceptance"]["per_unit_debit_absolute_tolerance_usd"])
    before_ms = int(contract["acceptance"]["funding_time_must_be_before_ms"])
    matches: list[dict[str, str]] = []
    for row in rows:
        try:
            rate = Decimal(str(row["fundingRate"]))
            mark = Decimal(str(row["markPrice"]))
            timestamp = int(row["fundingTime"])
        except (KeyError, ValueError, TypeError):
            continue
        debit = abs(rate) * mark
        if (
            row.get("symbol") == "GLWUSDT"
            and row.get("rateType") == "Special"
            and rate < 0
            and timestamp < before_ms
            and abs(debit - dividend) <= tolerance
        ):
            matches.append(
                {
                    "absolute_difference_usd": str(abs(debit - dividend)),
                    "fundingRate": str(row["fundingRate"]),
                    "fundingTime": str(timestamp),
                    "markPrice": str(row["markPrice"]),
                    "per_unit_debit_usd": str(debit),
                    "rateType": str(row["rateType"]),
                    "symbol": str(row["symbol"]),
                }
            )
    return matches, "exactly_one_match" if len(matches) == 1 else f"match_count_{len(matches)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    claimed_hash = contract.pop("contract_sha256")
    actual_hash = _sha256(_canonical(contract))
    if claimed_hash != actual_hash:
        raise SystemExit(f"contract hash mismatch: {claimed_hash} != {actual_hash}")
    contract["contract_sha256"] = claimed_hash

    history_receipt = _request(contract["history_request"], args.raw_dir)
    receipts = [history_receipt]
    matches: list[dict[str, str]] = []
    gate_status = "history_request_failed"
    history_rows: list[dict[str, Any]] = []
    try:
        loaded_history = _load_json(history_receipt)
        if isinstance(loaded_history, list):
            history_rows = [row for row in loaded_history if isinstance(row, dict)]
        matches, gate_status = _history_gate(contract, loaded_history)
    except (ValueError, json.JSONDecodeError):
        pass

    conditional_executed = len(matches) == 1
    if conditional_executed:
        with ThreadPoolExecutor(max_workers=len(contract["conditional_batch"])) as executor:
            futures = {
                executor.submit(_request, item, args.raw_dir): item
                for item in contract["conditional_batch"]
            }
            batch = [future.result() for future in as_completed(futures)]
        receipts.extend(sorted(batch, key=lambda item: item["label"]))

    journal = {
        "conditional_batch_executed": conditional_executed,
        "contract_sha256": claimed_hash,
        "gate_status": gate_status,
        "request_count": len(receipts),
        "receipts": receipts,
        "retry_count": 0,
        "schema_version": "binance-glw-special-funding-trigger-journal-v1",
    }
    journal_sha = _sha256(_canonical(journal))
    journal["journal_sha256"] = journal_sha
    _write_atomic(args.journal, _canonical(journal))

    request_success = all(item["status_code"] == 200 and not item["error"] for item in receipts)
    result = {
        "accepted_edge": False,
        "conditional_batch_executed": conditional_executed,
        "contract_sha256": claimed_hash,
        "deployment_ready": False,
        "gate_status": gate_status,
        "history_exact_matches": matches,
        "history_latest_row": history_rows[-1] if history_rows else None,
        "history_observed_rate_types": sorted(
            {str(row.get("rateType")) for row in history_rows}
        ),
        "history_row_count": len(history_rows),
        "history_special_row_count": sum(
            row.get("rateType") == "Special" for row in history_rows
        ),
        "journal_path": args.journal.as_posix(),
        "journal_sha256": journal_sha,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "public_net_distribution_floor_usd": "0",
        "next_retry_trigger": "after_2026_08_31T00_00_00Z_freeze_one_terminal_history_reconciliation_for_mechanism_evidence_only_no_2026_GLW_book_capture",
        "request_count": len(receipts),
        "request_success": request_success,
        "schema_version": "binance-glw-special-funding-trigger-result-v1",
        "status": (
            "exact_special_funding_gate_passed_conditional_public_batch_retained_economics_unadjudicated"
            if conditional_executed and request_success
            else "one_use_pre_snapshot_observation_consumed_no_matching_special_row_conditional_books_prohibited_future_event_unresolved"
        ),
        "trading_authority": False,
    }
    result["result_sha256"] = _sha256(_canonical(result))
    _write_atomic(args.result, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
