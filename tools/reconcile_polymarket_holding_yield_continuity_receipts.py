"""Reconcile exact retained Polymarket holding-yield rows to Polygon receipts."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
PUSD_TOKEN = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
YIELD_DISTRIBUTOR = "0x607c8c9866ef3b4665c5a384188706be738d8bf8"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


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


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _canonical_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _topic_address(topic: object) -> str:
    value = str(topic).lower()
    if len(value) != 66 or not value.startswith("0x"):
        raise ValueError("ERC-20 address topic is invalid")
    return "0x" + value[-40:]


def _payout_transfer(
    receipt: Mapping[str, object], *, wallet: str, amount: Decimal
) -> bool:
    for raw_log in _list(receipt.get("logs"), name="receipt logs"):
        log = _mapping(raw_log, name="receipt log")
        topics = _list(log.get("topics"), name="receipt log topics")
        if (
            str(log.get("address") or "").lower() == PUSD_TOKEN
            and len(topics) == 3
            and str(topics[0]).lower() == TRANSFER_TOPIC
            and _topic_address(topics[1]) == YIELD_DISTRIBUTOR
            and _topic_address(topics[2]) == wallet
            and Decimal(int(str(log.get("data")), 16)) / Decimal(1_000_000) == amount
        ):
            return True
    return False


def _write_journal(path: Path, journal: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(journal) + "\n").encode("ascii"))


def _validate_contract(contract: Mapping[str, object], contract_path: Path) -> None:
    if contract.get("schema_version") != (
        "polymarket-holding-yield-continuity-receipts-v8-contract"
    ):
        raise ValueError("contract schema differs")
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    frozen_at = str(contract.get("frozen_at_utc") or "")
    if not frozen_at.endswith(("Z", "+00:00")):
        raise ValueError("frozen_at_utc lacks an explicit UTC offset")
    frozen_ms = int(
        __import__("datetime")
        .datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
        .timestamp()
        * 1000
    )
    if frozen_ms > time.time_ns() // 1_000_000:
        raise ValueError("frozen_at_utc is in the future")
    implementation = _mapping(contract["implementation"], name="implementation")
    if Path(str(implementation["path"])) != Path(__file__).relative_to(ROOT):
        raise ValueError("implementation path differs")
    if _sha256(Path(__file__).read_bytes()) != implementation["sha256"]:
        raise ValueError("implementation hash mismatch")
    if contract_path != ROOT / str(contract["contract_path"]):
        raise ValueError("contract path differs")


def _load_source(source: Mapping[str, object]) -> object:
    path = ROOT / str(source["path"])
    payload = path.read_bytes()
    if _sha256(payload) != source["sha256"]:
        raise ValueError(f"retained source hash mismatch: {path}")
    return json.loads(payload)


def _validate_case(case: Mapping[str, object]) -> dict[str, object]:
    asset = str(case["asset"])
    wallet = str(case["wallet"]).lower()
    condition_id = str(case["condition_id"]).lower()
    shares = Decimal(str(case["shares_per_outcome"]))
    prior_timestamp = int(case["prior_yield_timestamp"])
    selected_timestamp = int(case["selected_yield_timestamp"])
    selected_amount = Decimal(str(case["selected_amount_pusd"]))
    selected_hash = str(case["selected_transaction_hash"]).lower()

    positions = [
        _mapping(row, name=f"{asset} position")
        for row in _list(_load_source(case["positions_source"]), name="positions")
    ]
    pair = [
        row
        for row in positions
        if str(row.get("conditionId") or "").lower() == condition_id
    ]
    if (
        len(pair) != 2
        or {str(row.get("outcome")) for row in pair} != {"Yes", "No"}
        or any(
            Decimal(str(row.get("size"))) != shares or row.get("mergeable") is not True
            for row in pair
        )
    ):
        raise ValueError(f"{asset} retained complete-set balance differs")

    activities = [
        _mapping(row, name=f"{asset} activity")
        for row in _list(_load_source(case["activity_source"]), name="activities")
    ]
    selected = [
        row
        for row in activities
        if int(row.get("timestamp", -1)) == selected_timestamp
        and str(row.get("transactionHash") or "").lower() == selected_hash
    ]
    if len(selected) != 1 or selected[0].get("type") != "YIELD":
        raise ValueError(f"{asset} exact retained YIELD row differs")
    if Decimal(str(selected[0].get("usdcSize"))) != selected_amount:
        raise ValueError(f"{asset} retained payout amount differs")
    interval = [
        row
        for row in activities
        if prior_timestamp < int(row.get("timestamp", -1)) <= selected_timestamp
    ]
    if len(interval) != 1 or interval[0] != selected[0]:
        raise ValueError(f"{asset} selected interval has another wallet activity")
    interval_seconds = selected_timestamp - prior_timestamp
    if not 82_800 <= interval_seconds <= 90_000:
        raise ValueError(f"{asset} selected interval is not daily")
    return {
        "asset": asset,
        "wallet": wallet,
        "condition_id": condition_id,
        "shares_per_outcome": format(shares, "f"),
        "prior_yield_timestamp": prior_timestamp,
        "selected_yield_timestamp": selected_timestamp,
        "selected_interval_seconds": interval_seconds,
        "selected_amount_pusd": format(selected_amount, "f"),
        "selected_transaction_hash": selected_hash,
        "retained_pair_equal_and_mergeable": True,
        "only_selected_yield_in_interval": True,
    }


def run(
    *, contract_path: Path, raw_dir: Path, journal_path: Path, output_path: Path
) -> dict[str, object]:
    if raw_dir.exists() or journal_path.exists() or output_path.exists():
        raise FileExistsError("one-use raw, journal, or output path already exists")
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract, contract_path)
    raw_dir.mkdir(parents=True, exist_ok=False)
    journal: dict[str, object] = {
        "schema_version": "polymarket-holding-yield-continuity-receipts-v8-journal",
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_sha256": contract["contract_sha256"],
        "requests": [],
    }
    _write_journal(journal_path, journal)
    cases = [_validate_case(_mapping(row, name="case")) for row in contract["cases"]]
    session = requests.Session()
    session.headers.update({"User-Agent": "simple-ai-trading-public-edge-research/1.0"})
    try:
        for request_id, case in enumerate(cases):
            body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "eth_getTransactionReceipt",
                "params": [case["selected_transaction_hash"]],
            }
            planned = {
                "request_id": request_id,
                "method": "POST",
                "url": contract["polygon_rpc_url"],
                "body_sha256": _sha256(_canonical_json(body).encode("ascii")),
                "state": "planned",
                "planned_at_ms": time.time_ns() // 1_000_000,
            }
            journal["requests"].append(planned)
            _write_journal(journal_path, journal)
            response = session.post(
                str(contract["polygon_rpc_url"]), json=body, timeout=30
            )
            payload = response.content
            raw_path = raw_dir / f"{str(case['asset']).lower()}-receipt.raw"
            write_bytes_atomic(raw_path, payload)
            planned.update(
                {
                    "state": "received",
                    "completed_at_ms": time.time_ns() // 1_000_000,
                    "status_code": response.status_code,
                    "response_bytes": len(payload),
                    "response_sha256": _sha256(payload),
                    "raw_path": raw_path.relative_to(ROOT).as_posix(),
                }
            )
            _write_journal(journal_path, journal)
            response.raise_for_status()
            envelope = _mapping(response.json(), name="RPC envelope")
            receipt = _mapping(envelope.get("result"), name="receipt")
            if (
                envelope.get("id") != request_id
                or envelope.get("error") is not None
                or receipt.get("status") != "0x1"
                or str(receipt.get("transactionHash") or "").lower()
                != case["selected_transaction_hash"]
                or not _payout_transfer(
                    receipt,
                    wallet=str(case["wallet"]),
                    amount=Decimal(str(case["selected_amount_pusd"])),
                )
            ):
                raise ValueError(f"{case['asset']} payout receipt does not reconcile")
            case["receipt"] = {
                "block_number": int(str(receipt["blockNumber"]), 16),
                "successful_exact_distributor_pusd_transfer": True,
                "source": {
                    key: planned[key]
                    for key in (
                        "method",
                        "url",
                        "status_code",
                        "response_bytes",
                        "response_sha256",
                        "raw_path",
                    )
                },
            }

        result: dict[str, object] = {
            "schema_version": "polymarket-holding-yield-continuity-receipts-v8",
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": "reconcile_two_additional_retained_daily_YIELD_rows_to_exact_public_Polygon_pUSD_transfers",
            "contract": {
                "path": contract["contract_path"],
                "sha256": contract["contract_sha256"],
            },
            "cases": cases,
            "adjudication": {
                "all_latest_retained_receipts_reconciled": True,
                "accepted_historical_scoped_edge_preserved": True,
                "current_rate_qualified_by_this_monitor": False,
                "deployment_ready": False,
                "future_profit_guaranteed": False,
                "public_profit_floor_for_new_capital_pusd": "0",
                "next_action": "continue_only_distinct_public_continuity_monitoring_or_wait_for_material_terms_or_economics_change",
                "status": "two_additional_daily_payout_receipts_reconciled",
            },
            "authority": contract["authority"],
            "implementation": contract["implementation"],
            "source_contract": contract["source_contract"],
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        write_bytes_atomic(
            output_path, (_canonical_json(result) + "\n").encode("ascii")
        )
        journal.update(
            {
                "state": "completed",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "result_sha256": result["result_sha256"],
            }
        )
        _write_journal(journal_path, journal)
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
        _write_journal(journal_path, journal)
        raise


def main() -> int:
    contract_path = ROOT / (
        "docs/model-research/polymarket/"
        "complete-set-holding-yield-continuity-receipts-contract-v8-2026-08-29.json"
    )
    raw_dir = ROOT / "data/polymarket-holding-yield-continuity-receipts-v8/raw"
    journal_path = ROOT / (
        "data/polymarket-holding-yield-continuity-receipts-v8/journal.json"
    )
    output_path = ROOT / (
        "docs/model-research/polymarket/"
        "complete-set-holding-yield-continuity-receipts-v8-2026-08-29.json"
    )
    result = run(
        contract_path=contract_path,
        raw_dir=raw_dir,
        journal_path=journal_path,
        output_path=output_path,
    )
    print(
        _canonical_json(
            {
                "case_count": len(result["cases"]),
                "request_count": len(result["cases"]),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
