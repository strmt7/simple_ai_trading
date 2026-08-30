"""Reconcile retained Binance funding estimates to later realized rates."""

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


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("timestamp must include an offset")
    return parsed.astimezone(UTC)


def _journal(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(
    *, name: str, url: str, raw_path: Path, journal_path: Path, timeout: int
) -> tuple[bytes, dict[str, Any]]:
    requested_at_ms = time.time_ns() // 1_000_000
    _journal(
        journal_path,
        {
            "method": "GET",
            "name": name,
            "phase": "intent",
            "request_body_sha256": _sha256(b""),
            "requested_at_ms": requested_at_ms,
            "url": url,
        },
    )
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
        "name": name,
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


def _walk_estimates(value: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {
            "symbol",
            "time",
            "nextFundingTime",
            "lastFundingRate",
        }.issubset(value):
            rows.append(value)
        for child in value.values():
            rows.extend(_walk_estimates(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_estimates(child))
    return rows


def _funding_rows(payload: bytes, *, symbol: str, limit: int) -> list[dict[str, Any]]:
    values = json.loads(payload)
    if not isinstance(values, list):
        raise RuntimeError("funding response must be an array")
    if len(values) >= limit:
        raise RuntimeError("funding response hit the frozen limit")
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict) or value.get("symbol") != symbol:
            raise RuntimeError("funding response escaped the frozen symbol")
        funding_time = int(value["fundingTime"])
        funding_rate = _decimal(value["fundingRate"], name="fundingRate")
        rows.append(
            {
                "symbol": symbol,
                "funding_time_ms": funding_time,
                "funding_rate": format(funding_rate, "f"),
                "rate_type": value.get("rateType"),
            }
        )
    times = [row["funding_time_ms"] for row in rows]
    if times != sorted(set(times)):
        raise RuntimeError("funding rows are not strictly ordered and unique")
    return rows


def _load_rows(path: Path, *, symbol: str) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    values = json.loads(payload)
    if not isinstance(values, list):
        raise RuntimeError("retained funding source must be an array")
    rows: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict) and value.get("symbol") == symbol:
            rows.append(
                {
                    "symbol": symbol,
                    "funding_time_ms": int(value["fundingTime"]),
                    "funding_rate": format(
                        _decimal(value["fundingRate"], name="fundingRate"), "f"
                    ),
                    "rate_type": value.get("rateType"),
                }
            )
    return rows


def run(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if _parse_utc(str(contract["frozen_at_utc"])) > datetime.now(UTC):
        raise RuntimeError("frozen_at_utc is in the future")

    sources: dict[str, Path] = {}
    for source in contract["retained_sources"]:
        path = _root_path(str(source["path"]))
        if _sha256(path.read_bytes()) != source["sha256"]:
            raise RuntimeError("retained source hash mismatch")
        sources[source["name"]] = path

    outputs = {
        name: _root_path(str(path)) for name, path in contract["outputs"].items()
    }
    for path in outputs.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)

    captured: dict[str, list[dict[str, Any]]] = {}
    receipts: list[dict[str, Any]] = []
    for request_contract in contract["requests"]:
        symbol = str(request_contract["symbol"])
        raw_path = outputs[str(request_contract["output_key"])]
        payload, receipt = _capture(
            name=str(request_contract["name"]),
            url=str(request_contract["url"]),
            raw_path=raw_path,
            journal_path=outputs["journal_path"],
            timeout=int(request_contract["timeout_seconds"]),
        )
        receipts.append(receipt)
        if receipt["status_code"] != 200:
            raise RuntimeError(f"unexpected HTTP status {receipt['status_code']}")
        rows = _funding_rows(
            payload, symbol=symbol, limit=int(request_contract["limit"])
        )
        if any(
            row["funding_time_ms"] < int(request_contract["start_time_ms"])
            or row["funding_time_ms"] > int(request_contract["end_time_ms"])
            for row in rows
        ):
            raise RuntimeError("funding row escaped the frozen interval")
        captured[symbol] = rows

    evaluations: list[dict[str, Any]] = []
    tolerance_ms = int(contract["decision"]["funding_time_tolerance_ms"])
    for episode in contract["episodes"]:
        snapshot = json.loads(sources[str(episode["snapshot_source"])].read_bytes())
        estimate_rows = _walk_estimates(snapshot)
        for expected in episode["estimates"]:
            symbol = str(expected["symbol"])
            matching_estimates = [
                row
                for row in estimate_rows
                if row.get("symbol") == symbol
                and int(row.get("time", -1)) == int(episode["snapshot_time_ms"])
                and int(row.get("nextFundingTime", -1))
                == int(episode["funding_time_ms"])
                and str(row.get("lastFundingRate")) == str(expected["estimate"])
            ]
            if len(matching_estimates) != 1:
                raise RuntimeError("retained estimate does not reconstruct uniquely")
            if episode["funding_source_kind"] == "captured":
                funding = captured[symbol]
            else:
                funding = _load_rows(
                    sources[str(expected["funding_source"])], symbol=symbol
                )
            matches = [
                row
                for row in funding
                if abs(row["funding_time_ms"] - int(episode["funding_time_ms"]))
                <= tolerance_ms
            ]
            if len(matches) != 1:
                raise RuntimeError("realized funding row does not reconstruct uniquely")
            actual = matches[0]
            estimate = Decimal(str(expected["estimate"]))
            realized = Decimal(actual["funding_rate"])
            evaluations.append(
                {
                    "episode": episode["name"],
                    "symbol": symbol,
                    "snapshot_time_ms": episode["snapshot_time_ms"],
                    "funding_time_ms": actual["funding_time_ms"],
                    "lead_seconds": (
                        int(episode["funding_time_ms"])
                        - int(episode["snapshot_time_ms"])
                    )
                    // 1000,
                    "estimate": format(estimate, "f"),
                    "realized": format(realized, "f"),
                    "error_bips": format((realized - estimate) * Decimal(10000), "f"),
                    "exact_match": estimate == realized,
                }
            )

    changed = [row for row in evaluations if not row["exact_match"]]
    status = (
        "terminal_estimate_not_locked_at_observed_lead_times"
        if changed
        else "all_observed_estimates_matched_but_general_lock_not_proved"
    )
    result: dict[str, Any] = {
        "schema_version": "binance-funding-estimate-lock-v1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "request_count": len(receipts),
            "receipts": receipts,
            "symbols": sorted(captured),
        },
        "evaluation": {
            "observation_count": len(evaluations),
            "exact_match_count": len(evaluations) - len(changed),
            "changed_count": len(changed),
            "rows": evaluations,
        },
        "adjudication": {
            "status": status,
            "known_at_entry_at_observed_lead_times_proved": False,
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "book_or_fee_request_permitted": False,
            "reason": "A displayed funding estimate is not a guaranteed transfer unless every applicable rule and final value are fixed at entry; even an exact historical match would not prove future locking or after-cost carry.",
            "next_action": "Do not trade or request books from these observations; reopen only on an official fixed-at-entry funding contract or a separately preregistered near-finality executable study whose conservative guaranteed transfer clears every cost.",
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
                "observations": result["evaluation"]["observation_count"],
                "changed": result["evaluation"]["changed_count"],
                "network_requests": result["capture"]["request_count"],
                "books_requested": 0,
                "credentials_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
