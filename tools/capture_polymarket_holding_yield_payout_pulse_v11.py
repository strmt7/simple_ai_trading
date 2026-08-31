"""Run one frozen public BTC holding-yield continuity pulse."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("ascii"))


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if (
        contract.get("schema_version")
        != "polymarket-holding-yield-payout-pulse-v11-contract"
    ):
        raise ValueError("contract schema differs")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise ValueError("contract path mismatch")
    if _canonical_hash(contract, field="contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    if _sha256(Path(__file__).read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    now = datetime.now(timezone.utc)
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    not_before = datetime.fromisoformat(
        str(contract["not_before_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or not_before.tzinfo is None or now < frozen:
        raise ValueError("contract timestamp is invalid or future")
    if now < not_before:
        raise ValueError("not-before gate is not satisfied")
    for parent in contract["parent_evidence"]:
        parent_path = _root_path(str(parent["path"]))
        payload = _mapping(json.loads(parent_path.read_bytes()), name="parent")
        field = str(parent["hash_field"])
        if payload.get(field) != parent["sha256"]:
            raise ValueError(f"parent declared hash mismatch: {parent['path']}")
        if _canonical_hash(payload, field=field) != parent["sha256"]:
            raise ValueError(f"parent canonical hash mismatch: {parent['path']}")


def run(contract_path: Path) -> dict[str, object]:
    contract = _mapping(json.loads(contract_path.read_bytes()), name="contract")
    _validate_contract(contract, contract_path)
    raw_path = _root_path(str(contract["outputs"]["raw_path"]))
    journal_path = _root_path(str(contract["outputs"]["journal_path"]))
    output_path = _root_path(str(contract["outputs"]["result_path"]))
    if raw_path.exists() or journal_path.exists() or output_path.exists():
        raise FileExistsError("one-use raw, journal, or output path already exists")

    journal: dict[str, object] = {
        "schema_version": "polymarket-holding-yield-payout-pulse-v11-journal",
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
            timeout=int(contract["request"]["timeout_seconds"]),
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
        normalized = [_mapping(row, name="activity row") for row in rows]
        if any(row.get("type") != "YIELD" for row in normalized):
            raise ValueError("type-filtered response contained a non-YIELD row")
        baseline = int(contract["baseline"]["selected_yield_timestamp"])
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
            "schema_version": "polymarket-holding-yield-payout-pulse-v11",
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": "one_request_post_gate_BTC_holding_yield_continuity_pulse",
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
                    else "no_new_BTC_YIELD_row_after_frozen_gate"
                ),
                "accepted_historical_scoped_edge_preserved": True,
                "current_rate_qualified": False,
                "deployment_ready": False,
                "future_profit_guaranteed": False,
                "public_profit_floor_for_new_capital_pusd": "0",
                "next_action": (
                    "freeze_one_exact_receipt_reconciliation_for_selected_row"
                    if new_rows
                    else "wait_for_material_terms_economics_cross_asset_payout_or_new_daily_window_change"
                ),
            },
            "authority": contract["authority"],
            "implementation": contract["implementation"],
            "result_sha256": "",
        }
        result["result_sha256"] = _canonical_hash(result, field="result_sha256")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    result = run(_root_path(args.contract))
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
