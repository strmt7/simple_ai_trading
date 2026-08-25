"""Audit historical Binance quarterly delivery/spot mismatch without trading."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.quarterly_delivery_basis as basis_module
from simple_ai_trading.quarterly_delivery_basis import (
    quarterly_delivery_basis_observation,
    stressed_after_hurdle_basis_bips,
)
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-delivery-basis-audit-contract-v1.json"
)
SCHEMA_VERSION = "binance-quarterly-delivery-basis-audit-v1"
FAILURE_SCHEMA_VERSION = "binance-quarterly-delivery-basis-audit-failure-v1"
FUTURES_URL = "https://fapi.binance.com/futures/data/delivery-price"
SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


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


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a positive finite decimal")
    return parsed


def _verified_json(
    path: Path, *, expected_result_hash: str | None = None
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="ascii"))
    expected = str(payload.get("result_sha256") or "")
    body = dict(payload)
    body.pop("result_sha256", None)
    actual = _sha256(_canonical_json(body).encode("ascii"))
    if expected != actual or (
        expected_result_hash is not None and expected != expected_result_hash
    ):
        raise ValueError(f"source artifact hash is invalid: {path.name}")
    return payload


def _get(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, object],
    ledger: list[dict[str, object]],
) -> object:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(url, params=params, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    decoded: object | None = None
    try:
        decoded = response.json()
    except requests.JSONDecodeError:
        pass
    entry: dict[str, object] = {
        "url": response.url,
        "status_code": response.status_code,
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
        "raw_response_sha256": _sha256(response.content),
        "decoded_payload": decoded,
        "canonical_payload_sha256": (
            None
            if decoded is None
            else _sha256(_canonical_json(decoded).encode("ascii"))
        ),
    }
    ledger.append(entry)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        entry["retry_after"] = retry_after
        raise RuntimeError(
            "Binance rate limit reached; stopped without retry"
            + (f"; Retry-After={retry_after}" if retry_after else "")
        )
    response.raise_for_status()
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("Binance response exceeded the bounded size")
    if decoded is None:
        raise ValueError(f"source {url} did not return JSON")
    return decoded


def _deliveries(
    raw: object,
    *,
    cutoff_ms: int,
    count: int,
) -> tuple[tuple[int, Decimal], ...]:
    deliveries: dict[int, Decimal] = {}
    for value in _list(raw, name="delivery prices"):
        row = _mapping(value, name="delivery price")
        timestamp = row.get("deliveryTime")
        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp <= 0
        ):
            raise ValueError("delivery time is invalid")
        price = _decimal(
            row.get("deliveryPrice"),
            name="delivery price",
            positive=True,
        )
        if timestamp in deliveries and deliveries[timestamp] != price:
            raise ValueError("duplicate delivery time has conflicting prices")
        if timestamp < cutoff_ms:
            deliveries[timestamp] = price
    selected = tuple(sorted(deliveries.items())[-count:])
    if len(selected) != count:
        raise ValueError("delivery source lacks the frozen completed sample")
    return selected


def _median(values: list[Decimal]) -> Decimal:
    selected = sorted(values)
    middle = len(selected) // 2
    if len(selected) % 2:
        return selected[middle]
    return (selected[middle - 1] + selected[middle]) / 2


def _current_carry(contract: Mapping[str, object]) -> dict[str, object]:
    source = _mapping(contract["source_binding"], name="source binding")
    path = ROOT / str(source["current_carry_result_path"])
    return _verified_json(
        path,
        expected_result_hash=str(source["current_carry_result_sha256"]),
    )


def run(
    *,
    session: requests.Session | None = None,
    ledger: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run the frozen historical audit and return a source-bound result."""

    source_ledger = [] if ledger is None else ledger
    contract = _verified_json(CONTRACT_PATH)
    if (
        contract.get("status")
        != "frozen_before_historical_delivery_or_spot_price_access"
    ):
        raise ValueError("delivery basis contract is not frozen")
    historical = _mapping(contract["historical_source"], name="historical source")
    cutoff_ms = int(historical["completed_delivery_cutoff_ms"])
    count = int(historical["completed_deliveries_per_pair"])
    http = session or requests.Session()
    started_ms = time.time_ns() // 1_000_000
    pair_deliveries: dict[str, tuple[tuple[int, Decimal], ...]] = {}
    for pair in historical["pairs"]:
        raw = _get(
            http,
            FUTURES_URL,
            params={"pair": pair},
            ledger=source_ledger,
        )
        pair_deliveries[str(pair)] = _deliveries(
            raw,
            cutoff_ms=cutoff_ms,
            count=count,
        )
    pair_results: list[dict[str, object]] = []
    worst_by_pair: dict[str, Decimal] = {}
    for pair in historical["pairs"]:
        observations: list[dict[str, object]] = []
        low_mismatches: list[Decimal] = []
        close_mismatches: list[Decimal] = []
        for delivery_time_ms, delivery_price in pair_deliveries[str(pair)]:
            raw = _get(
                http,
                SPOT_KLINES_URL,
                params={
                    "symbol": pair,
                    "interval": historical["spot_kline_interval"],
                    "startTime": delivery_time_ms,
                    "endTime": delivery_time_ms
                    + int(historical["spot_window_end_offset_ms"]),
                    "limit": int(historical["spot_kline_limit"]),
                },
                ledger=source_ledger,
            )
            bars = [
                _list(value, name="spot kline")
                for value in _list(raw, name="spot klines")
            ]
            observation = quarterly_delivery_basis_observation(
                delivery_time_ms=delivery_time_ms,
                delivery_price=delivery_price,
                spot_klines=bars,
            )
            low_mismatches.append(observation.minimum_low_mismatch_bips)
            close_mismatches.append(observation.fifth_close_mismatch_bips)
            observations.append(
                {
                    "delivery_time_ms": delivery_time_ms,
                    "delivery_price": str(delivery_price),
                    "post_delivery_minimum_low": str(
                        observation.post_delivery_minimum_low
                    ),
                    "post_delivery_fifth_close": str(
                        observation.post_delivery_fifth_close
                    ),
                    "minimum_low_mismatch_bips": str(
                        observation.minimum_low_mismatch_bips
                    ),
                    "fifth_close_mismatch_bips": str(
                        observation.fifth_close_mismatch_bips
                    ),
                }
            )
        worst = min(low_mismatches)
        worst_by_pair[str(pair)] = worst
        pair_results.append(
            {
                "pair": pair,
                "delivery_count": len(observations),
                "observations": observations,
                "minimum_low_mismatch_bips": str(worst),
                "median_minimum_low_mismatch_bips": str(_median(low_mismatches)),
                "median_fifth_close_mismatch_bips": str(_median(close_mismatches)),
            }
        )
    carry = _current_carry(contract)
    stressed_results: list[dict[str, object]] = []
    for screen in carry["screens"]:
        if screen["contract_type"] != "NEXT_QUARTER":
            continue
        pair = str(screen["pair"])
        mismatch = worst_by_pair[pair]
        for row in screen["quantity_results"]:
            after_hurdle = _decimal(
                row["after_hurdle_basis_bips"],
                name="after-hurdle basis",
            )
            stressed = stressed_after_hurdle_basis_bips(
                after_hurdle_basis_bips=after_hurdle,
                worst_observed_mismatch_bips=mismatch,
            )
            stressed_results.append(
                {
                    "pair": pair,
                    "symbol": screen["symbol"],
                    "quantity": row["quantity"],
                    "after_35_bps_hurdle_basis_bips": str(after_hurdle),
                    "worst_observed_delivery_spot_low_mismatch_bips": str(mismatch),
                    "stressed_basis_bips": str(stressed),
                    "stress_positive": stressed > 0,
                }
            )
    eligible = bool(stressed_results) and all(
        bool(row["stress_positive"]) for row in stressed_results
    )
    completed_ms = time.time_ns() // 1_000_000
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "historical_quarterly_delivery_price_vs_spot_exit_proxy_audit",
        "started_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "pair_results": pair_results,
        "current_next_quarter_stress": stressed_results,
        "source_contract": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": _sha256(CONTRACT_PATH.read_bytes()),
            "contract_result_sha256": contract["result_sha256"],
            "request_ledger": source_ledger,
            "implementation": {
                "tool_path": Path(__file__).relative_to(ROOT).as_posix(),
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(basis_module.__file__).relative_to(ROOT).as_posix(),
                "module_sha256": _sha256(Path(basis_module.__file__).read_bytes()),
            },
        },
        "verdict": {
            "status": (
                "collateral_and_liquidation_design_eligible_not_an_edge"
                if eligible
                else "rejected_delivery_basis_stress"
            ),
            "request_count": len(source_ledger),
            "all_next_quarter_sizes_stress_positive": eligible,
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
        "safety": {
            "public_data_only": True,
            "credentials_used": False,
            "orders_placed": False,
            "retry_count": 0,
        },
        "limitations": [
            "Historical one-minute trade lows are adverse price proxies, not executable bid depth or fill evidence.",
            "Historical delivery mismatch does not guarantee a future settlement or spot exit relationship.",
            "The 35 bps entry remains a sensitivity hurdle rather than authenticated account fee evidence.",
            "Quarterly futures commission, settlement charges, collateral opportunity cost, margin, liquidation, outages, and taxes remain unresolved.",
            "The current basis snapshot was post-selection and this audit cannot establish persistence.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    return artifact


def _failure_payload(
    *,
    error: Exception,
    ledger: list[dict[str, object]],
) -> dict[str, object]:
    contract = _verified_json(CONTRACT_PATH)
    artifact: dict[str, object] = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "terminal_failure_without_retry",
        "error": {"type": type(error).__name__, "message": str(error)},
        "source_contract": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": _sha256(CONTRACT_PATH.read_bytes()),
            "contract_result_sha256": contract["result_sha256"],
            "request_ledger": ledger,
            "implementation": {
                "tool_path": Path(__file__).relative_to(ROOT).as_posix(),
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(basis_module.__file__).relative_to(ROOT).as_posix(),
                "module_sha256": _sha256(Path(basis_module.__file__).read_bytes()),
            },
        },
        "authority": {
            "accepted_edge": False,
            "credentials_used": False,
            "orders_placed": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
    }
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger: list[dict[str, object]] = []
    try:
        result = run(ledger=ledger)
        exit_code = 0
    except Exception as exc:  # terminal evidence must precede CLI failure
        result = _failure_payload(error=exc, ledger=ledger)
        exit_code = 1
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result.get("verdict", result.get("error")), indent=2))
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
