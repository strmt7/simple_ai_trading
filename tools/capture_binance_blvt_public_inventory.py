from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-blvt-primary-market-nav-parity-contract-v1-2026-08-29.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-blvt-primary-market-nav-parity-public-gate-v1-2026-08-29.json"
)
DATA_ROOT = ROOT / "data/binance-blvt-public-inventory-v1"
RAW_PATH = DATA_ROOT / "raw/exchange-info.json"
JOURNAL_PATH = DATA_ROOT / "request-journal.jsonl"


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


def _journal(payload: dict[str, Any]) -> None:
    with JOURNAL_PATH.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(url: str) -> tuple[bytes, int]:
    requested_at_ms = time.time_ns() // 1_000_000
    intent = {
        "method": "GET",
        "name": "binance-public-exchange-info-blvt-inventory",
        "phase": "intent",
        "request_body_sha256": _sha256(b""),
        "requested_at_ms": requested_at_ms,
        "url": url,
    }
    _journal(intent)
    request = Request(
        url,
        method="GET",
        headers={"User-Agent": "simple-ai-trading-public-research/1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            response_bytes = response.read()
            status_code = response.status
    except HTTPError as exc:
        response_bytes = exc.read()
        status_code = exc.code
        RAW_PATH.write_bytes(response_bytes)
        _journal(
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
                "response_bytes": len(response_bytes),
                "response_sha256": _sha256(response_bytes),
                "status_code": status_code,
            }
        )
        raise
    RAW_PATH.write_bytes(response_bytes)
    _journal(
        {
            **intent,
            "completed_at_ms": time.time_ns() // 1_000_000,
            "phase": "completed",
            "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
            "response_bytes": len(response_bytes),
            "response_sha256": _sha256(response_bytes),
            "status_code": status_code,
        }
    )
    return response_bytes, status_code


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = contract["contract_sha256"]
    if _canonical_hash(contract, "contract_sha256") != contract_hash:
        raise RuntimeError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if DATA_ROOT.exists() or RESULT_PATH.exists():
        raise RuntimeError("one-use output already exists")
    RAW_PATH.parent.mkdir(parents=True)

    response_bytes, status_code = _capture(contract["capture"]["url"])
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")
    payload = json.loads(response_bytes)
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise RuntimeError("exchangeInfo symbols array is absent")

    leveraged: list[dict[str, Any]] = []
    for row in symbols:
        permissions = set(row.get("permissions", []))
        permission_sets = row.get("permissionSets", [])
        flattened_sets = {
            permission for group in permission_sets for permission in group
        }
        if "LEVERAGED" not in permissions | flattened_sets:
            continue
        leveraged.append(
            {
                "base_asset": row["baseAsset"],
                "quote_asset": row["quoteAsset"],
                "status": row["status"],
                "symbol": row["symbol"],
            }
        )
    leveraged.sort(key=lambda row: row["symbol"])
    trading = [row for row in leveraged if row["status"] == "TRADING"]

    result: dict[str, Any] = {
        "schema_version": "binance-blvt-primary-market-nav-parity-public-gate-v1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract_hash,
        },
        "authority": contract["authority"],
        "public_inventory": {
            "exchange_info_symbol_count": len(symbols),
            "leveraged_permission_symbol_count": len(leveraged),
            "trading_leveraged_symbol_count": len(trading),
            "trading_leveraged_symbols": trading,
            "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
            "raw_sha256": _sha256(response_bytes),
        },
        "adjudication": {
            "candidate_edge": bool(trading),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "next_action": (
                "only_with_an_ephemeral_API_key_and_explicit_GET_only_authority_"
                "capture_current_BLVT_tokenInfo_then_stop_before_books_unless_an_"
                "exact_NAV_fee_gap_clears_spot_hedge_funding_delay_and_stress"
                if trading
                else "terminalize_the_current_public_inventory_until_a_new_BLVT_listing"
            ),
        },
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(result["adjudication"], sort_keys=True))


if __name__ == "__main__":
    main()
