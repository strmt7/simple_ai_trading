"""Screen one frozen public wallet-day for causal opposite-leg complete-set locks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request


getcontext().prec = 40
SCHEMA = "polymarket-wallet-opposite-lock-result-v1"
EMPTY_SHA256 = _sha256(b"")
SLUG = re.compile(r"^(btc|eth|sol)-updown-(5m|15m|4h)-(\d+)$")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"invalid {label}") from exc
    if not parsed.is_finite():
        raise RuntimeError(f"non-finite {label}")
    return parsed


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an explicit UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")

    request = contract.get("request")
    if not isinstance(request, dict) or request.get("method") != "GET":
        raise RuntimeError("only one frozen public GET is supported")
    url = str(request.get("url") or "")
    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    expected_query = {
        "end": [str(contract["validation_window"]["end_epoch_exclusive"])],
        "limit": [str(contract["request"]["limit"])],
        "offset": ["0"],
        "start": [str(contract["validation_window"]["start_epoch_inclusive"])],
        "takerOnly": ["true"],
        "user": [contract["wallet"].lower()],
    }
    if (
        parsed.scheme != "https"
        or parsed.hostname != "data-api.polymarket.com"
        or parsed.path != "/trades"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or query != expected_query
        or request.get("count") != 1
        or request.get("body_sha256") != EMPTY_SHA256
    ):
        raise RuntimeError("request boundary changed")
    if contract.get("authority") != {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _all_in_cost(price: Decimal, fee_rate: Decimal) -> Decimal:
    return price + fee_rate * price * (Decimal(1) - price)


def _screen(contract: dict[str, Any], rows: list[Any]) -> dict[str, Any]:
    economics = contract["economics"]
    fee_rate = _decimal(economics["taker_fee_rate"], "taker fee rate")
    tick_size = _decimal(economics["tick_size"], "tick size")
    adverse_ticks = int(economics["hedge_adverse_ticks"])
    min_lag = int(economics["minimum_lag_seconds"])
    max_lag = int(economics["maximum_lag_seconds"])
    close_buffer = int(economics["minimum_seconds_before_close"])
    durations = {key: int(value) for key, value in economics["durations"].items()}
    start = int(contract["validation_window"]["start_epoch_inclusive"])
    end = int(contract["validation_window"]["end_epoch_exclusive"])
    wallet = contract["wallet"].lower()
    limit = int(contract["request"]["limit"])

    schema_errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            schema_errors.append(f"row_{index}_not_object")
            continue
        try:
            timestamp = int(row["timestamp"])
            outcome_index = int(row["outcomeIndex"])
            quantity = _decimal(row["size"], "size")
            price = _decimal(row["price"], "price")
            condition_id = str(row["conditionId"])
            event_slug = str(row["eventSlug"])
            side = str(row["side"]).upper()
            row_wallet = str(row["proxyWallet"]).lower()
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            schema_errors.append(f"row_{index}_{type(exc).__name__}")
            continue
        if (
            row_wallet != wallet
            or not (start <= timestamp < end)
            or outcome_index not in {0, 1}
            or quantity <= 0
            or not (Decimal(0) < price < Decimal(1))
            or side not in {"BUY", "SELL"}
            or not condition_id
            or not event_slug
        ):
            schema_errors.append(f"row_{index}_boundary")
            continue
        match = SLUG.fullmatch(event_slug)
        normalized.append(
            {
                "condition_id": condition_id,
                "duration": match.group(2) if match else None,
                "event_slug": event_slug,
                "event_start": int(match.group(3)) if match else None,
                "index": index,
                "outcome_index": outcome_index,
                "price": price,
                "quantity": quantity,
                "scope_asset": match.group(1) if match else None,
                "side": side,
                "timestamp": timestamp,
            }
        )

    complete_page = len(rows) < limit
    source_gate_passed = not schema_errors and complete_page
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if source_gate_passed:
        for row in normalized:
            if row["scope_asset"] is not None:
                grouped[row["condition_id"]].append(row)

    excluded_sell_conditions = {
        condition_id
        for condition_id, condition_rows in grouped.items()
        if any(row["side"] == "SELL" for row in condition_rows)
    }
    locks: list[dict[str, Any]] = []
    for condition_id, condition_rows in sorted(grouped.items()):
        if condition_id in excluded_sell_conditions:
            continue
        lots: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
        for row in sorted(
            condition_rows, key=lambda value: (value["timestamp"], value["index"])
        ):
            if row["side"] != "BUY":
                continue
            outcome = row["outcome_index"]
            opposite = 1 - outcome
            timestamp = row["timestamp"]
            lots[opposite] = [
                lot
                for lot in lots[opposite]
                if timestamp - lot["timestamp"] <= max_lag
                and lot["remaining"] > 0
            ]
            remaining = row["quantity"]
            stressed_price = min(
                Decimal(1), row["price"] + tick_size * adverse_ticks
            )
            stressed_hedge_cost = _all_in_cost(stressed_price, fee_rate)
            event_close = row["event_start"] + durations[row["duration"]]
            for lot in lots[opposite]:
                if remaining <= 0:
                    break
                lag = timestamp - lot["timestamp"]
                locked_per_share = Decimal(1) - lot["all_in_cost"] - stressed_hedge_cost
                if (
                    lag < min_lag
                    or lag > max_lag
                    or timestamp > event_close - close_buffer
                    or locked_per_share <= 0
                ):
                    continue
                matched = min(remaining, lot["remaining"])
                pnl = matched * locked_per_share
                locks.append(
                    {
                        "asset": row["scope_asset"],
                        "completion_timestamp": timestamp,
                        "condition_id": condition_id,
                        "duration": row["duration"],
                        "event_slug": row["event_slug"],
                        "first_leg_all_in_cost_per_share": str(lot["all_in_cost"]),
                        "hedge_observed_price": str(row["price"]),
                        "hedge_stressed_all_in_cost_per_share": str(
                            stressed_hedge_cost
                        ),
                        "lag_seconds": lag,
                        "locked_pnl": str(pnl),
                        "locked_pnl_per_share": str(locked_per_share),
                        "matched_shares": str(matched),
                    }
                )
                remaining -= matched
                lot["remaining"] -= matched
            lots[opposite] = [
                lot for lot in lots[opposite] if lot["remaining"] > 0
            ]
            if remaining > 0:
                lots[outcome].append(
                    {
                        "all_in_cost": _all_in_cost(row["price"], fee_rate),
                        "remaining": remaining,
                        "timestamp": timestamp,
                    }
                )

    total_pnl = sum((_decimal(row["locked_pnl"], "locked pnl") for row in locks), Decimal(0))
    total_shares = sum(
        (_decimal(row["matched_shares"], "matched shares") for row in locks),
        Decimal(0),
    )
    by_asset: dict[str, dict[str, Any]] = {}
    for asset in economics["assets"]:
        asset_locks = [row for row in locks if row["asset"] == asset]
        by_asset[asset] = {
            "condition_count": len({row["condition_id"] for row in asset_locks}),
            "lock_count": len(asset_locks),
            "locked_pnl": str(
                sum(
                    (
                        _decimal(row["locked_pnl"], "asset locked pnl")
                        for row in asset_locks
                    ),
                    Decimal(0),
                )
            ),
            "matched_shares": str(
                sum(
                    (
                        _decimal(row["matched_shares"], "asset matched shares")
                        for row in asset_locks
                    ),
                    Decimal(0),
                )
            ),
        }
    midpoint = start + (end - start) // 2
    halves: dict[str, dict[str, Any]] = {}
    for name, predicate in {
        "early": lambda timestamp: timestamp < midpoint,
        "late": lambda timestamp: timestamp >= midpoint,
    }.items():
        half_locks = [row for row in locks if predicate(row["completion_timestamp"])]
        halves[name] = {
            "condition_count": len({row["condition_id"] for row in half_locks}),
            "lock_count": len(half_locks),
            "locked_pnl": str(
                sum(
                    (
                        _decimal(row["locked_pnl"], "half locked pnl")
                        for row in half_locks
                    ),
                    Decimal(0),
                )
            ),
        }
    condition_pnl: dict[str, Decimal] = defaultdict(Decimal)
    for row in locks:
        condition_pnl[row["condition_id"]] += _decimal(
            row["locked_pnl"], "condition locked pnl"
        )
    maximum_condition_share = (
        max(condition_pnl.values()) / total_pnl
        if condition_pnl and total_pnl > 0
        else None
    )
    gates = contract["validation_gates"]
    gate_results = {
        "aggregate_locked_pnl_positive": total_pnl > 0,
        "complete_page": complete_page,
        "condition_count": len(condition_pnl) >= int(gates["minimum_conditions"]),
        "early_and_late_positive": all(
            _decimal(value["locked_pnl"], "half pnl") > 0
            and value["condition_count"] >= int(gates["minimum_conditions_per_half"])
            for value in halves.values()
        ),
        "lock_count": len(locks) >= int(gates["minimum_locks"]),
        "maximum_condition_concentration": maximum_condition_share is not None
        and maximum_condition_share
        <= _decimal(
            gates["maximum_single_condition_pnl_share"],
            "maximum condition concentration",
        ),
        "minimum_positive_assets": sum(
            _decimal(value["locked_pnl"], "asset pnl") > 0
            and value["condition_count"] > 0
            for value in by_asset.values()
        )
        >= int(gates["minimum_positive_assets"]),
        "schema": not schema_errors,
    }
    passed = source_gate_passed and all(gate_results.values())
    return {
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "historical_out_of_sample_candidate": passed,
            "profitability_claim": False,
            "public_forward_profit_floor": "0",
            "status": (
                "out_of_sample_historical_opposite_lock_candidate_passed"
                if passed
                else "out_of_sample_opposite_lock_candidate_rejected"
            ),
            "trading_authority": False,
        },
        "analysis": {
            "asset_results": by_asset,
            "condition_count": len(condition_pnl),
            "excluded_sell_condition_count": len(excluded_sell_conditions),
            "half_results": halves,
            "lock_count": len(locks),
            "locks": locks,
            "matched_shares": str(total_shares),
            "maximum_single_condition_pnl_share": (
                str(maximum_condition_share)
                if maximum_condition_share is not None
                else None
            ),
            "stress_locked_pnl_pusd": str(total_pnl),
        },
        "gate_results": gate_results,
        "source_gate": {
            "complete_page": complete_page,
            "normalized_row_count": len(normalized),
            "passed": source_gate_passed,
            "raw_row_count": len(rows),
            "schema_error_count": len(schema_errors),
            "schema_errors": schema_errors[:20],
            "scoped_condition_count": len(grouped),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    _validate_contract(contract, contract_path)
    paths = {
        name: _root_path(path) for name, path in contract["outputs"].items()
    }
    for path in paths.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
    raw, receipt = _request(
        method="GET",
        url=contract["request"]["url"],
        body=b"",
        name=contract["request_name"],
        raw_path=paths["raw_path"],
        raw_relative_path=contract["outputs"]["raw_path"],
        journal_path=paths["journal_path"],
    )
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise RuntimeError("trade response must be an array")
    screen = _screen(contract, decoded)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "response_byte_ceiling": contract["response_byte_ceiling"],
            "response_byte_ceiling_passed": len(raw)
            <= int(contract["response_byte_ceiling"]),
        },
        **screen,
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/screen_polymarket_wallet_opposite_lock.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    if not result["capture"]["response_byte_ceiling_passed"]:
        result["adjudication"] = {
            "accepted_edge": False,
            "deployment_ready": False,
            "historical_out_of_sample_candidate": False,
            "profitability_claim": False,
            "public_forward_profit_floor": "0",
            "status": "response_byte_ceiling_failed",
            "trading_authority": False,
        }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    paths["result_path"].write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "condition_count": result["analysis"]["condition_count"],
                "historical_candidate": result["adjudication"][
                    "historical_out_of_sample_candidate"
                ],
                "lock_count": result["analysis"]["lock_count"],
                "payloads_printed": 0,
                "response_bytes": len(raw),
                "response_sha256": receipt["response_sha256"],
                "source_gate_passed": result["source_gate"]["passed"],
                "stress_locked_pnl_pusd": result["analysis"][
                    "stress_locked_pnl_pusd"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
