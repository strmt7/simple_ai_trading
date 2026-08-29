"""Run the frozen one-use post-conflict Polymarket holding-yield reconciliation."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "polymarket-holding-yield-post-conflict-reconciliation-v7"
PUSD_TOKEN = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
YIELD_DISTRIBUTOR = "0x607c8c9866ef3b4665c5a384188706be738d8bf8"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
PAYOUT_SCALE = Decimal("0.0001")


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


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _embedded_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    observed = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != observed:
        raise ValueError(f"{field} mismatch: expected {claimed}, observed {observed}")
    return observed


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
            and Decimal(int(str(log.get("data")), 16)) / Decimal(1_000_000)
            == amount
        ):
            return True
    return False


class _Client:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "simple-ai-trading-public-edge-research/1.0"}
        )
        self.request_count = 0

    def request(
        self,
        method: str,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
        body: object | None = None,
    ) -> tuple[object, dict[str, object]]:
        response = None
        try:
            response = self.session.request(
                method, url, params=params, json=body, timeout=30
            )
            response.raise_for_status()
            payload = response.content
            raw_path = self.raw_dir / f"{name}.raw"
            write_bytes_atomic(raw_path, payload)
            self.request_count += 1
            source = {
                "method": method,
                "url": response.url,
                "status_code": response.status_code,
                "response_bytes": len(payload),
                "response_sha256": _sha256(payload),
                "raw_path": raw_path.as_posix(),
            }
            try:
                return response.json(), source
            except requests.JSONDecodeError as exc:
                raise ValueError(f"{name} did not return JSON") from exc
        finally:
            response = None


def _validate_contract(contract: Mapping[str, object], *, contract_path: Path) -> None:
    _embedded_hash(contract, "contract_result_sha256")
    if contract.get("schema_version") != f"{SCHEMA_VERSION}-contract":
        raise ValueError("contract schema differs")
    not_before_ms = int(contract["not_before_ms"])
    if time.time_ns() // 1_000_000 < not_before_ms:
        raise ValueError("frozen not-before boundary has not elapsed")
    implementation = _mapping(contract["implementation"], name="implementation")
    if implementation.get("path") != "tools/reconcile_polymarket_holding_yield_post_conflict.py":
        raise ValueError("contract implementation path differs")
    if _sha256(Path(__file__).read_bytes()) != implementation.get("sha256"):
        raise ValueError("implementation hash differs")
    if contract_path.as_posix() != str(contract["contract_path"]):
        raise ValueError("contract path differs")


def _candidate_matches(
    *, shares: Decimal, amount: Decimal, annual_rates: list[Decimal]
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for rate in annual_rates:
        hourly = shares * rate / Decimal(365) / Decimal(24)
        for sampled_hours in range(25):
            if (hourly * sampled_hours).quantize(
                PAYOUT_SCALE, rounding=ROUND_DOWN
            ) == amount:
                matches.append(
                    {
                        "annual_rate": _decimal_text(rate),
                        "sampled_hours": sampled_hours,
                    }
                )
    return matches


def run(
    *, contract_path: Path, raw_dir: Path, journal_path: Path, output_path: Path
) -> dict[str, object]:
    if output_path.exists() or journal_path.exists() or raw_dir.exists():
        raise FileExistsError("one-use output, journal, or raw directory already exists")
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract, contract_path=contract_path)
    raw_dir.mkdir(parents=True, exist_ok=False)
    journal: dict[str, object] = {
        "schema_version": f"{SCHEMA_VERSION}-journal",
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_result_sha256": contract["contract_result_sha256"],
        "request_count": 0,
    }
    write_bytes_atomic(
        journal_path, (_canonical_json(journal) + "\n").encode("ascii")
    )
    client = _Client(raw_dir)
    started_ms = time.time_ns() // 1_000_000
    try:
        conflict_snapshot_s = int(contract["conflict_snapshot_ms"]) // 1000
        annual_rates = [
            Decimal(str(value)) for value in _list(contract["candidate_annual_rates"], name="candidate rates")
        ]
        cases: list[dict[str, object]] = []
        selected_rows: list[tuple[str, str, Decimal, dict[str, object]]] = []
        for raw_case in _list(contract["cases"], name="cases"):
            case = _mapping(raw_case, name="case")
            asset = str(case["asset"])
            wallet = str(case["wallet"]).lower()
            condition = str(case["condition_id"]).lower()
            shares = Decimal(str(case["shares_per_outcome"]))
            positions_raw, positions_source = client.request(
                "GET",
                str(contract["data_api_positions_url"]),
                name=f"{asset.lower()}-positions",
                params={"user": wallet, "limit": 500, "offset": 0, "sizeThreshold": 0},
            )
            positions = [
                _mapping(row, name=f"{asset} position")
                for row in _list(positions_raw, name=f"{asset} positions")
            ]
            if len(positions) >= 500 or any(
                str(row.get("proxyWallet") or "").lower() != wallet for row in positions
            ):
                raise ValueError(f"{asset} positions are incomplete or cross-wallet")
            pair = [
                row
                for row in positions
                if str(row.get("conditionId") or "").lower() == condition
            ]
            if len(pair) != 2 or {str(row.get("outcome")) for row in pair} != {"Yes", "No"}:
                raise ValueError(f"{asset} selected pair differs")
            if any(
                Decimal(str(row.get("size"))) != shares or row.get("mergeable") is not True
                for row in pair
            ):
                raise ValueError(f"{asset} selected balances changed")
            if sum(
                (Decimal(str(row.get("currentValue"))) for row in pair), Decimal(0)
            ) != shares:
                raise ValueError(f"{asset} complete-set current value differs")

            activity_raw, activity_source = client.request(
                "GET",
                str(contract["data_api_activity_url"]),
                name=f"{asset.lower()}-activity",
                params={"user": wallet, "limit": 500, "offset": 0},
            )
            activities = [
                _mapping(row, name=f"{asset} activity")
                for row in _list(activity_raw, name=f"{asset} activities")
            ]
            if len(activities) >= 500:
                raise ValueError(f"{asset} activity reached its page limit")
            post_conflict_yield = sorted(
                (
                    row
                    for row in activities
                    if row.get("type") == "YIELD"
                    and int(row.get("timestamp", -1)) > conflict_snapshot_s
                ),
                key=lambda row: int(row["timestamp"]),
            )
            if len(post_conflict_yield) < 2:
                raise ValueError(f"{asset} lacks a wholly post-conflict interval")
            prior_row, selected = post_conflict_yield[0], post_conflict_yield[1]
            prior_timestamp = int(prior_row["timestamp"])
            selected_timestamp = int(selected["timestamp"])
            interval_seconds = selected_timestamp - prior_timestamp
            if not 82_800 <= interval_seconds <= 90_000:
                raise ValueError(f"{asset} selected distribution interval is not daily")
            if any(
                row.get("type") != "YIELD"
                and prior_timestamp <= int(row.get("timestamp", -1)) <= selected_timestamp
                for row in activities
            ):
                raise ValueError(f"{asset} wallet changed during the selected interval")
            amount = Decimal(str(selected["usdcSize"]))
            matches = _candidate_matches(
                shares=shares, amount=amount, annual_rates=annual_rates
            )
            tx_hash = str(selected.get("transactionHash") or "").lower()
            if len(tx_hash) != 66 or not tx_hash.startswith("0x"):
                raise ValueError(f"{asset} selected transaction hash is invalid")
            selected_rows.append((asset, wallet, amount, selected))
            cases.append(
                {
                    "asset": asset,
                    "wallet": wallet,
                    "condition_id": condition,
                    "shares_per_outcome": _decimal_text(shares),
                    "current_position_row_count": len(positions),
                    "selected_pair_unchanged_equal_and_mergeable": True,
                    "first_post_conflict_distribution_timestamp": prior_timestamp,
                    "selected_wholly_post_conflict_distribution_timestamp": selected_timestamp,
                    "selected_interval_seconds": interval_seconds,
                    "selected_amount_pusd": _decimal_text(amount),
                    "candidate_rate_sample_matches": matches,
                    "no_non_yield_activity_during_selected_interval": True,
                    "sources": {
                        "positions": positions_source,
                        "activity": activity_source,
                    },
                }
            )

        for index, ((asset, wallet, amount, selected), case_result) in enumerate(
            zip(selected_rows, cases, strict=True)
        ):
            rpc_raw, rpc_source = client.request(
                "POST",
                str(contract["polygon_rpc_url"]),
                name=f"{asset.lower()}-selected-receipt",
                body={
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "eth_getTransactionReceipt",
                    "params": [str(selected["transactionHash"])],
                },
            )
            envelope = _mapping(rpc_raw, name=f"{asset} RPC envelope")
            receipt = _mapping(envelope.get("result"), name=f"{asset} receipt")
            tx_hash = str(selected["transactionHash"]).lower()
            if (
                envelope.get("id") != index
                or envelope.get("error") is not None
                or receipt.get("status") != "0x1"
                or str(receipt.get("transactionHash") or "").lower() != tx_hash
                or not _payout_transfer(receipt, wallet=wallet, amount=amount)
            ):
                raise ValueError(f"{asset} selected payout receipt does not reconcile")
            case_result["receipt_reconciliation"] = {
                "transaction_hash": tx_hash,
                "block_number": int(str(receipt["blockNumber"]), 16),
                "successful_exact_pusd_transfer": True,
                "source": rpc_source,
            }

        unique_rates = {
            str(match["annual_rate"])
            for case in cases
            for match in case["candidate_rate_sample_matches"]
        }
        all_unique = all(
            len(case["candidate_rate_sample_matches"]) == 1 for case in cases
        )
        resolved_rate = next(iter(unique_rates)) if all_unique and len(unique_rates) == 1 else None
        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_ms": time.time_ns() // 1_000_000,
            "capture_started_at_ms": started_ms,
            "purpose": "one_use_public_reconciliation_of_the_first_daily_distribution_wholly_postdating_the_current_rate_conflict",
            "authority": {
                "public_unauthenticated_read_only": True,
                "credentials_used": False,
                "funds_used": False,
                "orders_or_transactions_submitted": 0,
                "trading_authority": False,
            },
            "contract": {
                "path": contract_path.as_posix(),
                "contract_result_sha256": contract["contract_result_sha256"],
            },
            "request_budget": {
                "planned_get_requests": 6,
                "planned_rpc_receipt_requests": 3,
                "actual_requests": client.request_count,
            },
            "cases": cases,
            "adjudication": {
                "candidate_rates_all_uniquely_resolved": all_unique,
                "resolved_annual_rate": resolved_rate,
                "current_operating_rate_qualified": resolved_rate is not None,
                "historical_scoped_edge_preserved": True,
                "deployment_ready": False,
                "future_profit_guaranteed": False,
                "public_profit_floor_for_new_capital_pusd": "0",
                "status": (
                    "post_conflict_realized_rate_resolved"
                    if resolved_rate is not None
                    else "post_conflict_rate_remains_unresolved"
                ),
            },
            "limitations": [
                "Public wallets prove the payout mechanism and current rate, not owned-account eligibility or external costs.",
                "The program rate is variable and future persistence is not guaranteed.",
                "No split, merge, order, transfer, account, or funded action was authorized or attempted.",
            ],
            "implementation": contract["implementation"],
            "source_contracts": contract["source_contracts"],
        }
        result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
        write_bytes_atomic(
            output_path, (_canonical_json(result) + "\n").encode("ascii")
        )
        journal.update(
            {
                "state": "completed",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "request_count": client.request_count,
                "result_sha256": result["result_sha256"],
            }
        )
        write_bytes_atomic(
            journal_path, (_canonical_json(journal) + "\n").encode("ascii")
        )
        return result
    except Exception as exc:
        journal.update(
            {
                "state": "failed",
                "failed_at_ms": time.time_ns() // 1_000_000,
                "request_count": client.request_count,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_bytes_atomic(
            journal_path, (_canonical_json(journal) + "\n").encode("ascii")
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        contract_path=args.contract,
        raw_dir=args.raw_dir,
        journal_path=args.journal,
        output_path=args.output,
    )
    print(json.dumps(result["adjudication"], indent=2))
    print(f"request_count={result['request_budget']['actual_requests']}")
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
