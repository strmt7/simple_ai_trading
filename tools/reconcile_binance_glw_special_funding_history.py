"""Reconcile the terminal GLW special-funding history after its snapshot."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("path escapes repository root")
    return path


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("timestamp must include an offset")
    return parsed.astimezone(UTC)


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise RuntimeError(f"{name} must be a finite decimal")
    return parsed


def _journal(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(
    *, url: str, raw_path: Path, journal_path: Path, timeout: int
) -> tuple[bytes, dict[str, Any]]:
    requested_at_ms = time.time_ns() // 1_000_000
    intent = {
        "method": "GET",
        "name": "GLWUSDT-terminal-special-funding-history-delta",
        "phase": "intent",
        "request_body_sha256": _sha256(b""),
        "requested_at_ms": requested_at_ms,
        "url": url,
    }
    _journal(journal_path, intent)
    request = Request(
        url,
        method="GET",
        headers={"User-Agent": "simple-ai-trading-public-research/1"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
            status_code = response.status
            content_type = response.headers.get("Content-Type", "")
            final_url = response.geturl()
    except HTTPError as exc:
        payload = exc.read()
        status_code = exc.code
        content_type = exc.headers.get("Content-Type", "")
        final_url = exc.geturl()
    raw_path.write_bytes(payload)
    receipt = {
        "completed_at_ms": time.time_ns() // 1_000_000,
        "content_type": content_type,
        "final_url": final_url,
        "method": "GET",
        "phase": "completed",
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "requested_at_ms": requested_at_ms,
        "response_bytes": len(payload),
        "response_sha256": _sha256(payload),
        "status_code": status_code,
        "url": url,
    }
    _journal(journal_path, receipt)
    return payload, receipt


def _funding_row(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("funding row must be an object")
    if value.get("symbol") != "GLWUSDT":
        raise RuntimeError("funding response escaped GLWUSDT")
    funding_time = int(value["fundingTime"])
    funding_rate = _decimal(value["fundingRate"], name="fundingRate")
    mark_price = _decimal(value["markPrice"], name="markPrice")
    rate_type = value.get("rateType")
    if not (
        funding_time > 0 and mark_price > 0 and rate_type in {"Regular", "Special"}
    ):
        raise RuntimeError("funding row has invalid fields")
    debit = abs(funding_rate) * mark_price
    return {
        "symbol": "GLWUSDT",
        "funding_time_ms": funding_time,
        "funding_time_utc": datetime.fromtimestamp(funding_time / 1000, UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "funding_rate": format(funding_rate, "f"),
        "mark_price": format(mark_price, "f"),
        "rate_type": rate_type,
        "per_unit_debit_usdt": format(debit, "f"),
    }


def _status(
    *,
    rows: list[dict[str, Any]],
    snapshot_ms: int,
    gross_dividend: Decimal,
    tolerance: Decimal,
) -> tuple[str, list[dict[str, Any]]]:
    special_negative = [
        row
        for row in rows
        if row["rate_type"] == "Special" and Decimal(row["funding_rate"]) < 0
    ]
    for row in special_negative:
        difference = abs(Decimal(row["per_unit_debit_usdt"]) - gross_dividend)
        row["absolute_difference_from_gross_dividend_usdt"] = format(difference, "f")
        row["matches_gross_dividend_tolerance"] = difference <= tolerance
        row["timing_relative_to_bstock_snapshot"] = (
            "before" if row["funding_time_ms"] < snapshot_ms else "at_or_after"
        )
    matching_before = [
        row
        for row in special_negative
        if row["matches_gross_dividend_tolerance"]
        and row["funding_time_ms"] < snapshot_ms
    ]
    matching_after = [
        row
        for row in special_negative
        if row["matches_gross_dividend_tolerance"]
        and row["funding_time_ms"] >= snapshot_ms
    ]
    if not rows:
        status = "terminal_no_new_history_rows"
    elif not special_negative:
        status = "terminal_no_negative_special_funding_row"
    elif matching_before:
        status = "matching_pre_snapshot_special_row_observed_mechanism_only"
    elif matching_after:
        status = (
            "terminal_matching_special_row_at_or_after_snapshot_no_pre_snapshot_gap"
        )
    else:
        status = "terminal_negative_special_row_magnitude_mismatch"
    return status, special_negative


def run(contract_path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    implementation_path = _root_path(str(contract["implementation"]["path"]))
    if (
        _sha256(implementation_path.read_bytes())
        != contract["implementation"]["sha256"]
    ):
        raise RuntimeError("implementation hash mismatch")

    observed_now = datetime.now(UTC) if now is None else now.astimezone(UTC)
    if _parse_utc(contract["frozen_at_utc"]) > observed_now:
        raise RuntimeError("frozen_at_utc is in the future")
    if observed_now < _parse_utc(contract["not_before_utc"]):
        raise RuntimeError("terminal reconciliation not-before gate is not satisfied")

    for source in contract["retained_sources"]:
        source_path = _root_path(str(source["path"]))
        if _sha256(source_path.read_bytes()) != source["sha256"]:
            raise RuntimeError("retained source hash mismatch")

    outputs = {
        name: _root_path(str(path)) for name, path in contract["outputs"].items()
    }
    for path in outputs.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)

    payload, receipt = _capture(
        url=contract["request"]["url"],
        raw_path=outputs["raw_path"],
        journal_path=outputs["journal_path"],
        timeout=int(contract["request"]["timeout_seconds"]),
    )
    if receipt["status_code"] != 200:
        raise RuntimeError(f"unexpected HTTP status {receipt['status_code']}")
    values = json.loads(payload)
    if not isinstance(values, list):
        raise RuntimeError("funding response must be an array")
    if len(values) >= int(contract["request"]["limit"]):
        raise RuntimeError("funding response hit the frozen limit")
    rows = [_funding_row(value) for value in values]
    times = [row["funding_time_ms"] for row in rows]
    if times != sorted(set(times)):
        raise RuntimeError("funding rows are not strictly ordered and unique")
    boundary = contract["decision"]
    if any(
        row["funding_time_ms"] < boundary["delta_start_time_ms"]
        or row["funding_time_ms"] > boundary["delta_end_time_ms"]
        for row in rows
    ):
        raise RuntimeError("funding row escaped the frozen delta interval")

    status, special_negative = _status(
        rows=rows,
        snapshot_ms=int(boundary["bstock_snapshot_time_ms"]),
        gross_dividend=Decimal(boundary["gross_dividend_per_share_usd"]),
        tolerance=Decimal(boundary["absolute_tolerance_usd"]),
    )
    result: dict[str, Any] = {
        "schema_version": "binance-glw-special-funding-terminal-reconciliation-v2",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "request_count": 1,
            "receipt": receipt,
            "response_row_count": len(rows),
            "response_complete_below_limit": len(rows)
            < int(contract["request"]["limit"]),
        },
        "history": {
            "prior_retained_row_count": boundary["prior_retained_row_count"],
            "delta_rows": rows,
            "delta_row_count": len(rows),
            "combined_row_count": boundary["prior_retained_row_count"] + len(rows),
            "negative_special_rows": special_negative,
            "negative_special_row_count": len(special_negative),
        },
        "adjudication": {
            "status": status,
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "public_profit_floor_usd": "0",
            "book_capture_permitted": False,
            "reason": "Terminal history can establish adjustment timing only; the conservative net bStock distribution floor and complete execution costs remain unproved.",
            "next_action": "Reopen only for a future independent weekend or holiday dividend episode with a new prospective contract and positive source-bound net distribution floor.",
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    outputs["result_path"].write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract)
    print(
        json.dumps(
            {
                "status": result["adjudication"]["status"],
                "delta_rows": result["history"]["delta_row_count"],
                "negative_special_rows": result["history"][
                    "negative_special_row_count"
                ],
                "network_requests": result["capture"]["request_count"],
                "books_requested": 0,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
