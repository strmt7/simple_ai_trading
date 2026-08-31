"""Reconcile one frozen V11 holding-yield row to its public Polygon receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic
from tools import reconcile_polymarket_holding_yield_payout_receipt as base


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
        != "polymarket-holding-yield-payout-receipt-v12-contract"
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
    parent = _mapping(contract["parent_result"], name="parent result")
    parent_path = _root_path(str(parent["path"]))
    payload = _mapping(json.loads(parent_path.read_bytes()), name="parent payload")
    if payload.get("result_sha256") != parent["result_sha256"]:
        raise ValueError("parent declared hash mismatch")
    if _canonical_hash(payload, field="result_sha256") != parent["result_sha256"]:
        raise ValueError("parent canonical hash mismatch")
    selected = payload["observation"]["first_new_yield_row"]
    if selected is None:
        raise ValueError("parent selected no payout row")
    if (
        selected["transaction_hash"] != contract["transaction_hash"]
        or selected["timestamp"] != contract["payout"]["selected_timestamp"]
        or selected["amount_pusd"] != contract["payout"]["amount_pusd"]
    ):
        raise ValueError("frozen payout differs from parent selection")


def run(contract_path: Path) -> dict[str, object]:
    contract = _mapping(json.loads(contract_path.read_bytes()), name="contract")
    _validate_contract(contract, contract_path)
    raw_path = _root_path(str(contract["outputs"]["raw_path"]))
    journal_path = _root_path(str(contract["outputs"]["journal_path"]))
    output_path = _root_path(str(contract["outputs"]["result_path"]))
    if raw_path.exists() or journal_path.exists() or output_path.exists():
        raise FileExistsError("one-use raw, journal, or output path already exists")
    body = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "eth_getTransactionReceipt",
        "params": [contract["transaction_hash"]],
    }
    journal: dict[str, object] = {
        "schema_version": "polymarket-holding-yield-payout-receipt-v12-journal",
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_sha256": contract["contract_sha256"],
        "request": {
            "method": "POST",
            "url": contract["request"]["url"],
            "body_sha256": _sha256(_canonical_json(body).encode("ascii")),
            "state": "planned",
            "planned_at_ms": time.time_ns() // 1_000_000,
        },
    }
    _write_json(journal_path, journal)
    try:
        response = requests.post(
            str(contract["request"]["url"]),
            json=body,
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
        envelope = _mapping(response.json(), name="RPC envelope")
        receipt = _mapping(envelope.get("result"), name="receipt")
        reconciled = (
            envelope.get("id") == 0
            and envelope.get("error") is None
            and receipt.get("status") == "0x1"
            and str(receipt.get("transactionHash") or "").lower()
            == contract["transaction_hash"]
            and base._exact_transfer(receipt, contract)
        )
        if not reconciled:
            raise ValueError("frozen payout does not reconcile to an exact transfer")
        result: dict[str, object] = {
            "schema_version": "polymarket-holding-yield-payout-receipt-v12",
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": "reconcile_one_V11_BTC_holding_yield_row_to_exact_public_Polygon_pUSD_transfer",
            "contract": {
                "path": contract["contract_path"],
                "sha256": contract["contract_sha256"],
            },
            "parent_result": contract["parent_result"],
            "payout": contract["payout"],
            "receipt": {
                "block_number": int(str(receipt["blockNumber"]), 16),
                "successful_exact_distributor_pusd_transfer": True,
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
                "status": "new_daily_BTC_payout_receipt_reconciled",
                "accepted_historical_scoped_edge_preserved": True,
                "current_single_wallet_payout_continuity_observed": True,
                "current_three_wallet_rate_qualified": False,
                "deployment_ready": False,
                "future_profit_guaranteed": False,
                "public_profit_floor_for_new_capital_pusd": "0",
                "next_action": "wait_for_material_terms_economics_or_cross_asset_payout_change",
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
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
