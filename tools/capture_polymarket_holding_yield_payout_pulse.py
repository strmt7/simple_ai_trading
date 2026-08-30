"""Run one frozen public activity request for a new holding-yield payout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]


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


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("ascii"))


def run(
    *, contract_path: Path, raw_path: Path, journal_path: Path, output_path: Path
) -> dict[str, object]:
    if raw_path.exists() or journal_path.exists() or output_path.exists():
        raise FileExistsError("one-use raw, journal, or output path already exists")
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    if contract.get("schema_version") != (
        "polymarket-holding-yield-payout-pulse-v9-contract"
    ):
        raise ValueError("contract schema differs")
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    if _sha256(Path(__file__).read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    frozen_at = str(contract.get("frozen_at_utc") or "")
    frozen_ms = int(
        __import__("datetime")
        .datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
        .timestamp()
        * 1000
    )
    if frozen_ms > time.time_ns() // 1_000_000:
        raise ValueError("frozen_at_utc is in the future")

    journal: dict[str, object] = {
        "schema_version": "polymarket-holding-yield-payout-pulse-v9-journal",
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_sha256": contract["contract_sha256"],
        "request": {
            "method": "GET",
            "url": contract["request"]["url"],
            "state": "planned",
            "planned_at_ms": time.time_ns() // 1_000_000,
        },
    }
    _write_json(journal_path, journal)
    try:
        response = requests.get(
            str(contract["request"]["url"]),
            headers={"User-Agent": "simple-ai-trading-public-edge-research/1.0"},
            timeout=30,
        )
        write_bytes_atomic(raw_path, response.content)
        journal["request"].update(
            {
                "state": "received",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "status_code": response.status_code,
                "response_bytes": len(response.content),
                "response_sha256": _sha256(response.content),
                "raw_path": raw_path.relative_to(ROOT).as_posix(),
            }
        )
        _write_json(journal_path, journal)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("activity response must be a list")
        baseline = int(contract["baseline"]["selected_yield_timestamp"])
        normalized = [_mapping(row, name="activity row") for row in rows]
        if any(row.get("type") != "YIELD" for row in normalized):
            raise ValueError("type-filtered response contained a non-YIELD row")
        new_rows = sorted(
            (row for row in normalized if int(row.get("timestamp", -1)) > baseline),
            key=lambda row: int(row["timestamp"]),
        )
        selected = None
        if new_rows:
            row = new_rows[0]
            selected = {
                "timestamp": int(row["timestamp"]),
                "interval_seconds": int(row["timestamp"]) - baseline,
                "amount_pusd": str(row.get("usdcSize")),
                "transaction_hash": str(row.get("transactionHash") or "").lower(),
            }
        result: dict[str, object] = {
            "schema_version": "polymarket-holding-yield-payout-pulse-v9",
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": "one_request_post_window_BTC_holding_yield_payout_pulse",
            "contract": {
                "path": contract["contract_path"],
                "sha256": contract["contract_sha256"],
            },
            "baseline": contract["baseline"],
            "observation": {
                "returned_row_count": len(normalized),
                "new_yield_row_count": len(new_rows),
                "first_new_yield_row": selected,
                "source": {
                    key: journal["request"][key]
                    for key in (
                        "method",
                        "url",
                        "status_code",
                        "response_bytes",
                        "response_sha256",
                        "raw_path",
                    )
                },
            },
            "adjudication": {
                "status": (
                    "new_BTC_YIELD_row_found_receipt_not_yet_reconciled"
                    if new_rows
                    else "no_new_BTC_YIELD_row_after_frozen_25h_window"
                ),
                "accepted_historical_scoped_edge_preserved": True,
                "current_rate_qualified": False,
                "deployment_ready": False,
                "future_profit_guaranteed": False,
                "public_profit_floor_for_new_capital_pusd": "0",
                "next_action": (
                    "freeze_one_exact_receipt_reconciliation_for_selected_row"
                    if new_rows
                    else "do_not_poll_again_before_2026-08-31T02:15:30Z_unless_official_terms_change"
                ),
            },
            "authority": contract["authority"],
            "implementation": contract["implementation"],
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        _write_json(output_path, result)
        journal.update(
            {
                "state": "completed",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "result_sha256": result["result_sha256"],
            }
        )
        _write_json(journal_path, journal)
        return result
    except Exception as exc:
        journal.update(
            {
                "state": "failed",
                "failed_at_ms": time.time_ns() // 1_000_000,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(journal_path, journal)
        raise


def main() -> int:
    contract_path = ROOT / (
        "docs/model-research/polymarket/"
        "complete-set-holding-yield-payout-pulse-contract-v9-2026-08-30.json"
    )
    raw_path = ROOT / (
        "data/polymarket-holding-yield-payout-pulse-v9/raw/btc-activity.raw"
    )
    journal_path = ROOT / (
        "data/polymarket-holding-yield-payout-pulse-v9/journal.json"
    )
    output_path = ROOT / (
        "docs/model-research/polymarket/"
        "complete-set-holding-yield-payout-pulse-v9-2026-08-30.json"
    )
    result = run(
        contract_path=contract_path,
        raw_path=raw_path,
        journal_path=journal_path,
        output_path=output_path,
    )
    print(
        _canonical_json(
            {
                "status": result["adjudication"]["status"],
                "new_yield_row_count": result["observation"]["new_yield_row_count"],
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
