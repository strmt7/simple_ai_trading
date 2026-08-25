"""Audit a frozen Binance quarterly pre-delivery unwind without trading."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.quarterly_pre_delivery_unwind as unwind_module
from simple_ai_trading.quarterly_pre_delivery_unwind import (
    pre_delivery_unwind_observation,
    stressed_pre_delivery_basis_bips,
)
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-pre-delivery-unwind-contract-v1.json"
)
SCHEMA_VERSION = "binance-quarterly-pre-delivery-unwind-audit-v1"
FAILURE_SCHEMA_VERSION = "binance-quarterly-pre-delivery-unwind-failure-v1"
FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
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


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _verified_json(
    path: Path,
    *,
    expected_result_hash: str | None = None,
    expected_raw_hash: str | None = None,
) -> dict[str, object]:
    raw = path.read_bytes()
    if expected_raw_hash is not None and _sha256(raw) != expected_raw_hash:
        raise ValueError(f"source artifact bytes are invalid: {path.name}")
    payload = json.loads(raw)
    expected = str(payload.get("result_sha256") or "")
    body = dict(payload)
    body.pop("result_sha256", None)
    actual = _sha256(_canonical_json(body).encode("ascii"))
    if expected != actual or (
        expected_result_hash is not None and expected != expected_result_hash
    ):
        raise ValueError(f"source artifact hash is invalid: {path.name}")
    return payload


def _contract() -> dict[str, object]:
    contract = _verified_json(CONTRACT_PATH)
    if (
        contract.get("status")
        != "frozen_before_historical_spot_or_futures_kline_access"
    ):
        raise ValueError("pre-delivery unwind contract is not frozen")
    return contract


def _bound_source(
    contract: Mapping[str, object],
    *,
    path_field: str,
    raw_hash_field: str,
    result_hash_field: str,
) -> dict[str, object]:
    source = _mapping(contract["source_binding"], name="source binding")
    return _verified_json(
        ROOT / str(source[path_field]),
        expected_raw_hash=str(source[raw_hash_field]),
        expected_result_hash=str(source[result_hash_field]),
    )


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


def _bars(raw: object, *, name: str) -> list[list[object]]:
    return [
        _list(row, name=f"{name} kline") for row in _list(raw, name=f"{name} klines")
    ]


def _median(values: list[Decimal]) -> Decimal:
    selected = sorted(values)
    middle = len(selected) // 2
    if len(selected) % 2:
        return selected[middle]
    return (selected[middle - 1] + selected[middle]) / 2


def _implementation() -> dict[str, object]:
    return {
        "tool_path": Path(__file__).relative_to(ROOT).as_posix(),
        "tool_sha256": _sha256(Path(__file__).read_bytes()),
        "module_path": Path(unwind_module.__file__).relative_to(ROOT).as_posix(),
        "module_sha256": _sha256(Path(unwind_module.__file__).read_bytes()),
    }


def run(
    *,
    session: requests.Session | None = None,
    ledger: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run the frozen 16-contract audit and return a source-bound result."""

    source_ledger = [] if ledger is None else ledger
    contract = _contract()
    _bound_source(
        contract,
        path_field="timestamp_adjudication_path",
        raw_hash_field="timestamp_adjudication_raw_file_sha256",
        result_hash_field="timestamp_adjudication_result_sha256",
    )
    _bound_source(
        contract,
        path_field="timing_semantics_path",
        raw_hash_field="timing_semantics_raw_file_sha256",
        result_hash_field="timing_semantics_result_sha256",
    )
    carry = _bound_source(
        contract,
        path_field="current_carry_path",
        raw_hash_field="current_carry_raw_file_sha256",
        result_hash_field="current_carry_result_sha256",
    )
    request_contract = _mapping(contract["request_contract"], name="request contract")
    metric = _mapping(contract["metric_contract"], name="metric contract")
    primary_horizon = int(metric["primary_horizon_minutes"])
    horizons = (
        primary_horizon,
        *(int(value) for value in metric["diagnostic_horizons_minutes"]),
    )
    http = session or requests.Session()
    started_ms = time.time_ns() // 1_000_000
    observations: list[dict[str, object]] = []
    for frozen_value in _list(contract["frozen_contracts"], name="frozen contracts"):
        frozen = _mapping(frozen_value, name="frozen contract")
        pair = str(frozen["pair"])
        symbol = str(frozen["symbol"])
        delivery_ms = int(frozen["scheduled_delivery_ms"])
        start_ms = delivery_ms + int(request_contract["window_start_offset_ms"])
        end_ms = delivery_ms + int(request_contract["window_end_offset_ms"])
        common = {
            "interval": request_contract["interval"],
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": int(request_contract["window_limit"]),
        }
        futures_raw = _get(
            http,
            FUTURES_KLINES_URL,
            params={"symbol": symbol, **common},
            ledger=source_ledger,
        )
        spot_raw = _get(
            http,
            SPOT_KLINES_URL,
            params={"symbol": pair, **common},
            ledger=source_ledger,
        )
        observation = pre_delivery_unwind_observation(
            scheduled_delivery_ms=delivery_ms,
            futures_klines=_bars(futures_raw, name="futures"),
            spot_klines=_bars(spot_raw, name="spot"),
            horizons_minutes=horizons,
        )
        observations.append(
            {
                "pair": pair,
                "symbol": symbol,
                "scheduled_delivery_ms": delivery_ms,
                "futures_last_bar_open_ms": observation.futures_last_bar_open_ms,
                "futures_last_bar_close_ms": observation.futures_last_bar_close_ms,
                "cutoff_consistent_with_normal_schedule": True,
                "horizons": [
                    {
                        "horizon_minutes": row.horizon_minutes,
                        "bar_open_time_ms": row.bar_open_time_ms,
                        "spot_low": str(row.spot_low),
                        "spot_close": str(row.spot_close),
                        "future_high": str(row.future_high),
                        "future_close": str(row.future_close),
                        "adverse_exit_basis_bips": str(row.adverse_exit_basis_bips),
                        "close_basis_bips": str(row.close_basis_bips),
                    }
                    for row in observation.horizon_observations
                ],
            }
        )
    maximum_primary_by_pair: dict[str, Decimal] = {}
    pair_results: list[dict[str, object]] = []
    for pair in ("BTCUSDT", "ETHUSDT"):
        selected = [row for row in observations if row["pair"] == pair]
        horizon_summaries: list[dict[str, object]] = []
        for horizon in horizons:
            values = [
                _decimal(
                    next(
                        value["adverse_exit_basis_bips"]
                        for value in row["horizons"]
                        if value["horizon_minutes"] == horizon
                    ),
                    name="adverse exit basis",
                )
                for row in selected
            ]
            maximum = max(values)
            if horizon == primary_horizon:
                maximum_primary_by_pair[pair] = maximum
            horizon_summaries.append(
                {
                    "horizon_minutes": horizon,
                    "observation_count": len(values),
                    "maximum_adverse_exit_basis_bips": str(maximum),
                    "median_adverse_exit_basis_bips": str(_median(values)),
                }
            )
        pair_results.append(
            {
                "pair": pair,
                "contract_count": len(selected),
                "primary_horizon_minutes": primary_horizon,
                "primary_maximum_adverse_exit_basis_bips": str(
                    maximum_primary_by_pair[pair]
                ),
                "horizon_summaries": horizon_summaries,
            }
        )
    stressed_results: list[dict[str, object]] = []
    for screen_value in carry["screens"]:
        screen = _mapping(screen_value, name="carry screen")
        if screen["contract_type"] != "NEXT_QUARTER":
            continue
        pair = str(screen["pair"])
        stress = maximum_primary_by_pair[pair]
        for row_value in screen["quantity_results"]:
            row = _mapping(row_value, name="carry quantity")
            after_hurdle = _decimal(
                row["after_hurdle_basis_bips"], name="after-hurdle basis"
            )
            stressed = stressed_pre_delivery_basis_bips(
                after_hurdle_basis_bips=after_hurdle,
                adverse_exit_basis_bips=stress,
            )
            stressed_results.append(
                {
                    "pair": pair,
                    "symbol": screen["symbol"],
                    "quantity": row["quantity"],
                    "after_35_bps_hurdle_basis_bips": str(after_hurdle),
                    "primary_historical_adverse_exit_basis_bips": str(stress),
                    "stressed_basis_bips": str(stressed),
                    "stress_positive": stressed > 0,
                }
            )
    eligible = len(stressed_results) == 6 and all(
        bool(row["stress_positive"]) for row in stressed_results
    )
    completed_ms = time.time_ns() // 1_000_000
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "fixed_pre_delivery_quarterly_cash_and_carry_unwind_stress",
        "started_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "observations": observations,
        "pair_results": pair_results,
        "current_next_quarter_stress": stressed_results,
        "source_contract": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": _sha256(CONTRACT_PATH.read_bytes()),
            "contract_result_sha256": contract["result_sha256"],
            "request_ledger": source_ledger,
            "implementation": _implementation(),
        },
        "verdict": {
            "status": (
                "authenticated_cost_collateral_and_execution_design_eligible_not_an_edge"
                if eligible
                else "rejected_primary_pre_delivery_basis_stress"
            ),
            "request_count": len(source_ledger),
            "all_current_next_quarter_sizes_primary_stress_positive": eligible,
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
            "One-minute trade highs and lows are adverse proxies, not synchronized executable depth or fill evidence.",
            "A final 07:59 trade bar plus no later bar is consistent with the normal cutoff but does not replace an authenticated order-state record.",
            "The 35 bps entry remains a sensitivity hurdle rather than authenticated account fee evidence.",
            "Quarterly futures commissions, settlement fallback, collateral opportunity cost, margin, liquidation, outages, and taxes remain unresolved.",
            "The current basis snapshot was post-selection and this audit cannot establish future persistence or capacity.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    return artifact


def _terminal_payload(
    *,
    schema_version: str,
    status: str,
    ledger: list[dict[str, object]],
    error: Exception | None = None,
) -> dict[str, object]:
    contract = _contract()
    artifact: dict[str, object] = {
        "schema_version": schema_version,
        "status": status,
        "source_contract": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": _sha256(CONTRACT_PATH.read_bytes()),
            "contract_result_sha256": contract["result_sha256"],
            "request_ledger": ledger,
            "implementation": _implementation(),
        },
        "authority": {
            "accepted_edge": False,
            "credentials_used": False,
            "orders_placed": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
    }
    if error is not None:
        artifact["error"] = {"type": type(error).__name__, "message": str(error)}
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    return artifact


def _reserved_payload() -> dict[str, object]:
    return _terminal_payload(
        schema_version=FAILURE_SCHEMA_VERSION,
        status="reserved_before_public_requests",
        ledger=[],
    )


def _failure_payload(
    *, error: Exception, ledger: list[dict[str, object]]
) -> dict[str, object]:
    return _terminal_payload(
        schema_version=FAILURE_SCHEMA_VERSION,
        status="terminal_failure_without_retry",
        ledger=ledger,
        error=error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite one-use output: {args.output}")
        return 2
    reserved = _reserved_payload()
    write_bytes_atomic(args.output, (_canonical_json(reserved) + "\n").encode("ascii"))
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
