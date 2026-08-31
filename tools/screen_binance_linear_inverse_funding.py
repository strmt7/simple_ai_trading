"""Run one frozen public linear-versus-inverse Binance funding preflight."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
ZERO = Decimal(0)
TEN_THOUSAND = Decimal(10_000)


def _canonical_hash(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _append_journal(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()


def _request_json(
    *, url: str, raw_path: Path, journal_path: Path, timeout_seconds: int
) -> tuple[Any, dict[str, Any]]:
    requested_at = _utc_now()
    started = time.monotonic()
    _append_journal(
        journal_path,
        {
            "event": "request_started",
            "method": "GET",
            "requested_at_utc": requested_at,
            "url": url,
        },
    )
    request = Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "User-Agent": "simple-ai-trading-rd/1"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
            status = int(response.status)
            headers = dict(response.headers.items())
    except HTTPError as exc:
        payload = exc.read()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)
        _append_journal(
            journal_path,
            {
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "event": "request_failed",
                "http_status": int(exc.code),
                "payload_bytes": len(payload),
                "payload_sha256": _sha256(payload),
                "received_at_utc": _utc_now(),
                "url": url,
            },
        )
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        _append_journal(
            journal_path,
            {
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "error_type": type(exc).__name__,
                "event": "request_failed",
                "received_at_utc": _utc_now(),
                "url": url,
            },
        )
        raise RuntimeError(f"request failed for {url}") from exc
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(payload)
    if status != 200:
        raise RuntimeError(f"unexpected HTTP {status} for {url}")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response for {url}") from exc
    receipt = {
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "http_date": headers.get("Date"),
        "http_status": status,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256(payload),
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "received_at_utc": _utc_now(),
        "url": url,
    }
    _append_journal(journal_path, {"event": "request_completed", **receipt})
    return parsed, receipt


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    expected_path = ROOT / contract["contract_path"]
    if contract_path != expected_path.resolve():
        raise RuntimeError("contract path mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    source = ROOT / contract["official_source"]["path"]
    if _sha256(source.read_bytes()) != contract["official_source"]["sha256"]:
        raise RuntimeError("official source hash mismatch")
    output = ROOT / contract["output_path"]
    raw_dir = ROOT / contract["retention"]["raw_directory"]
    journal = ROOT / contract["retention"]["journal_path"]
    if output.exists() or raw_dir.exists() or journal.exists():
        raise RuntimeError("one-use output boundary already exists")


def _eligible_symbols(
    dapi: dict[str, Any], fapi: dict[str, Any], assets: list[str]
) -> list[dict[str, Any]]:
    coin_rows = {
        row.get("baseAsset"): row
        for row in dapi.get("symbols", [])
        if row.get("contractType") == "PERPETUAL"
        and row.get("contractStatus") == "TRADING"
        and row.get("baseAsset") in assets
        and row.get("quoteAsset") == "USD"
        and row.get("marginAsset") == row.get("baseAsset")
    }
    linear_rows = {
        row.get("baseAsset"): row
        for row in fapi.get("symbols", [])
        if row.get("contractType") == "PERPETUAL"
        and row.get("status") == "TRADING"
        and row.get("baseAsset") in assets
        and row.get("quoteAsset") == "USDT"
        and row.get("marginAsset") == "USDT"
    }
    pairs: list[dict[str, Any]] = []
    for asset in assets:
        coin = coin_rows.get(asset)
        linear = linear_rows.get(asset)
        if coin is None or linear is None:
            continue
        pairs.append(
            {
                "asset": asset,
                "coin_contract_size_usd": str(coin.get("contractSize")),
                "coin_margin_asset": coin.get("marginAsset"),
                "coin_symbol": coin.get("symbol"),
                "linear_margin_asset": linear.get("marginAsset"),
                "linear_symbol": linear.get("symbol"),
            }
        )
    return pairs


def _bucket_rows(rows: Any, symbol: str, interval_ms: int) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"empty funding history for {symbol}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise RuntimeError(f"unexpected funding row for {symbol}")
        if "fundingTime" not in row or "fundingRate" not in row or "markPrice" not in row:
            raise RuntimeError(f"incomplete funding row for {symbol}")
        timestamp = int(row["fundingTime"])
        bucket = (timestamp + interval_ms // 2) // interval_ms
        if bucket in result:
            raise RuntimeError(f"duplicate funding bucket for {symbol}")
        result[bucket] = row
    return result


def _drawdown(values: list[Decimal]) -> Decimal:
    cumulative = ZERO
    peak = ZERO
    maximum = ZERO
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _role_result(
    rows: list[dict[str, Any]], *, orientation_sign: Decimal, execution_bips: Decimal
) -> dict[str, Any]:
    increments = [orientation_sign * row["raw_spread_bips"] for row in rows]
    gross = sum(increments, ZERO)
    round_trip = execution_bips * Decimal(4)
    net = gross - round_trip
    months: dict[str, list[Decimal]] = defaultdict(list)
    for row, increment in zip(rows, increments, strict=True):
        stamp = datetime.fromtimestamp(row["funding_time_ms"] / 1000, tz=timezone.utc)
        months[stamp.strftime("%Y-%m")].append(increment)
    month_rows = [
        {
            "gross_funding_bips": str(sum(values, ZERO)),
            "interval_count": len(values),
            "month": month,
            "positive": sum(values, ZERO) > ZERO,
        }
        for month, values in sorted(months.items())
    ]
    return {
        "all_observed_months_gross_positive": all(row["positive"] for row in month_rows),
        "entry_exit_execution_hurdle_bips": str(round_trip),
        "gross_funding_bips": str(gross),
        "interval_count": len(rows),
        "maximum_gross_drawdown_bips": str(_drawdown(increments)),
        "negative_interval_count": sum(value < ZERO for value in increments),
        "net_after_entry_exit_hurdle_bips": str(net),
        "observed_months": month_rows,
        "positive_interval_count": sum(value > ZERO for value in increments),
        "zero_interval_count": sum(value == ZERO for value in increments),
    }


def _evaluate_pair(
    pair: dict[str, Any], coin_rows: Any, linear_rows: Any, contract: dict[str, Any]
) -> dict[str, Any]:
    interval_ms = int(contract["population"]["funding_interval_hours"]) * 3_600_000
    coin = _bucket_rows(coin_rows, pair["coin_symbol"], interval_ms)
    linear = _bucket_rows(linear_rows, pair["linear_symbol"], interval_ms)
    common = sorted(set(coin) & set(linear))
    aligned: list[dict[str, Any]] = []
    maximum_skew = 0
    for bucket in common:
        coin_row = coin[bucket]
        linear_row = linear[bucket]
        skew = abs(int(coin_row["fundingTime"]) - int(linear_row["fundingTime"]))
        maximum_skew = max(maximum_skew, skew)
        aligned.append(
            {
                "coin_mark_price": Decimal(str(coin_row["markPrice"])),
                "funding_time_ms": max(
                    int(coin_row["fundingTime"]), int(linear_row["fundingTime"])
                ),
                "linear_mark_price": Decimal(str(linear_row["markPrice"])),
                "raw_spread_bips": (
                    Decimal(str(coin_row["fundingRate"]))
                    - Decimal(str(linear_row["fundingRate"]))
                )
                * TEN_THOUSAND,
            }
        )
    minimum = int(contract["population"]["minimum_aligned_rows"])
    if len(aligned) < minimum:
        return {
            **pair,
            "aligned_row_count": len(aligned),
            "failure_reasons": ["insufficient_aligned_funding_history"],
            "qualified_public_preflight": False,
        }
    if maximum_skew > int(contract["population"]["maximum_cross_venue_skew_ms"]):
        raise RuntimeError(f"funding timestamp skew exceeded for {pair['asset']}")
    training_stop = len(aligned) * 3 // 5
    validation_stop = len(aligned) * 4 // 5
    roles = {
        "training": aligned[:training_stop],
        "validation": aligned[training_stop:validation_stop],
        "test": aligned[validation_stop:],
    }
    training_raw = sum((row["raw_spread_bips"] for row in roles["training"]), ZERO)
    orientation_sign = Decimal(1) if training_raw >= ZERO else Decimal(-1)
    orientation = (
        "short_COIN_M_long_USDS_M"
        if orientation_sign > ZERO
        else "long_COIN_M_short_USDS_M"
    )
    execution = Decimal(contract["economics"]["one_way_execution_bips"])
    results = {
        name: _role_result(rows, orientation_sign=orientation_sign, execution_bips=execution)
        for name, rows in roles.items()
    }
    required = contract["gates"]["required_out_of_sample_roles"]
    failures: list[str] = []
    for role in required:
        if Decimal(results[role]["net_after_entry_exit_hurdle_bips"]) <= ZERO:
            failures.append(f"{role}_net_nonpositive")
        if not results[role]["all_observed_months_gross_positive"]:
            failures.append(f"{role}_month_persistence_failed")
        if Decimal(results[role]["maximum_gross_drawdown_bips"]) > Decimal(
            contract["gates"]["maximum_gross_drawdown_bips"]
        ):
            failures.append(f"{role}_drawdown_failed")
    return {
        **pair,
        "aligned_end_time_ms": aligned[-1]["funding_time_ms"],
        "aligned_row_count": len(aligned),
        "aligned_start_time_ms": aligned[0]["funding_time_ms"],
        "coin_source_row_count": len(coin_rows),
        "failure_reasons": failures,
        "fixed_training_selected_orientation": orientation,
        "linear_source_row_count": len(linear_rows),
        "maximum_timestamp_skew_ms": maximum_skew,
        "qualified_public_preflight": not failures,
        "role_results": results,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_bytes())
    if not isinstance(contract, dict):
        raise RuntimeError("contract must be an object")
    _validate_contract(contract, contract_path)

    raw_dir = ROOT / contract["retention"]["raw_directory"]
    journal_path = ROOT / contract["retention"]["journal_path"]
    output_path = ROOT / contract["output_path"]
    journal_path.parent.mkdir(parents=True, exist_ok=False)
    receipts: list[dict[str, Any]] = []
    try:
        dapi, receipt = _request_json(
            url=contract["requests"]["coin_exchange_info"],
            raw_path=raw_dir / "coin-exchange-info.json",
            journal_path=journal_path,
            timeout_seconds=int(contract["request_policy"]["timeout_seconds"]),
        )
        receipts.append(receipt)
        fapi, receipt = _request_json(
            url=contract["requests"]["linear_exchange_info"],
            raw_path=raw_dir / "linear-exchange-info.json",
            journal_path=journal_path,
            timeout_seconds=int(contract["request_policy"]["timeout_seconds"]),
        )
        receipts.append(receipt)
        if not isinstance(dapi, dict) or not isinstance(fapi, dict):
            raise RuntimeError("exchange information must be objects")
        pairs = _eligible_symbols(dapi, fapi, contract["population"]["assets"])
        pair_results: list[dict[str, Any]] = []
        for pair in pairs:
            coin_url = contract["requests"]["coin_history_template"].format(
                symbol=pair["coin_symbol"]
            )
            linear_url = contract["requests"]["linear_history_template"].format(
                symbol=pair["linear_symbol"]
            )
            coin_rows, receipt = _request_json(
                url=coin_url,
                raw_path=raw_dir / f"{pair['asset'].lower()}-coin-funding.json",
                journal_path=journal_path,
                timeout_seconds=int(contract["request_policy"]["timeout_seconds"]),
            )
            receipts.append(receipt)
            linear_rows, receipt = _request_json(
                url=linear_url,
                raw_path=raw_dir / f"{pair['asset'].lower()}-linear-funding.json",
                journal_path=journal_path,
                timeout_seconds=int(contract["request_policy"]["timeout_seconds"]),
            )
            receipts.append(receipt)
            pair_results.append(_evaluate_pair(pair, coin_rows, linear_rows, contract))
        qualified = sum(row["qualified_public_preflight"] for row in pair_results)
        result: dict[str, Any] = {
            "authority": contract["authority"],
            "completed_at_utc": _utc_now(),
            "contract": {
                "contract_sha256": contract["contract_sha256"],
                "path": contract["contract_path"],
            },
            "pair_results": pair_results,
            "request_receipts": receipts,
            "result_sha256": "",
            "schema_version": "binance-linear-inverse-funding-preflight-v1",
            "verdict": {
                "accepted_edge": False,
                "eligible_pair_count": len(pair_results),
                "profitability_claim": False,
                "qualified_public_preflight_count": qualified,
                "status": (
                    "public_persistence_candidate_requires_collateral_fee_basis_and_cross_regime_evidence"
                    if qualified
                    else "rejected_public_linear_inverse_funding_persistence_preflight"
                ),
                "trading_authority": False,
            },
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        _write_json(output_path, result)
        _append_journal(
            journal_path,
            {
                "event": "run_completed",
                "output_path": output_path.relative_to(ROOT).as_posix(),
                "result_sha256": result["result_sha256"],
                "timestamp_utc": _utc_now(),
            },
        )
    except Exception as exc:
        failure = {
            "authority": contract["authority"],
            "completed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "request_receipts": receipts,
            "result_sha256": "",
            "schema_version": "binance-linear-inverse-funding-preflight-failure-v1",
            "verdict": {
                "accepted_edge": False,
                "profitability_claim": False,
                "status": "terminal_one_use_capture_failure",
                "trading_authority": False,
            },
        }
        failure["result_sha256"] = _canonical_hash(failure, "result_sha256")
        _write_json(output_path, failure)
        _append_journal(
            journal_path,
            {
                "error_type": type(exc).__name__,
                "event": "run_failed",
                "result_sha256": failure["result_sha256"],
                "timestamp_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
