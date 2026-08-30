"""Reject or advance the TSMB dividend/perpetual hedge from retained funding."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-tsm-bstock-dividend-underdebit-contract-v1-2026-08-30.json"
)
OUTPUT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-tsm-bstock-dividend-underdebit-v1-2026-08-30.json"
)
RAW_DIR = ROOT / "data/binance-tsm-bstock-dividend-underdebit-v1/raw"
JOURNAL_PATH = ROOT / "data/binance-tsm-bstock-dividend-underdebit-v1/journal.json"


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


def _canonical_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("ascii"))


def _load_bound_json(source: Mapping[str, object]) -> object:
    path = ROOT / str(source["path"])
    payload = path.read_bytes()
    if _sha256(payload) != source["sha256"]:
        raise ValueError(f"retained source hash mismatch: {path}")
    return json.loads(payload)


def _validate_contract(contract: Mapping[str, object]) -> None:
    if contract.get("schema_version") != (
        "binance-tsm-bstock-dividend-underdebit-v1-contract"
    ):
        raise ValueError("contract schema differs")
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    implementation = _mapping(contract["implementation"], name="implementation")
    if _sha256(Path(__file__).read_bytes()) != implementation["sha256"]:
        raise ValueError("implementation hash mismatch")
    frozen_at = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise ValueError("frozen_at_utc lacks an explicit offset")
    if frozen_at.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise ValueError("frozen_at_utc is in the future")


def _capture_sources(
    contract: Mapping[str, object], journal: dict[str, object]
) -> list[dict[str, object]]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        }
    )
    receipts: list[dict[str, object]] = []
    for request_id, raw_source in enumerate(contract["issuer_sources"]):
        source = _mapping(raw_source, name="issuer source")
        request = {
            "request_id": request_id,
            "method": "GET",
            "url": source["url"],
            "state": "planned",
            "planned_at_ms": time.time_ns() // 1_000_000,
        }
        journal["requests"].append(request)
        _write_json(JOURNAL_PATH, journal)
        response = session.get(str(source["url"]), timeout=30)
        raw_path = RAW_DIR / f"{source['name']}.raw.html"
        write_bytes_atomic(raw_path, response.content)
        request.update(
            {
                "state": "received",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "status_code": response.status_code,
                "response_bytes": len(response.content),
                "response_sha256": _sha256(response.content),
                "raw_path": raw_path.relative_to(ROOT).as_posix(),
            }
        )
        _write_json(JOURNAL_PATH, journal)
        response.raise_for_status()
        text = response.content.decode("utf-8")
        missing = [value for value in source["required_text"] if value not in text]
        if missing:
            raise ValueError(f"{source['name']} required text missing: {missing}")
        receipts.append(
            {
                key: request[key]
                for key in (
                    "method",
                    "url",
                    "status_code",
                    "response_bytes",
                    "response_sha256",
                    "raw_path",
                )
            }
        )
    return receipts


def run() -> dict[str, object]:
    if RAW_DIR.exists() or JOURNAL_PATH.exists() or OUTPUT_PATH.exists():
        raise FileExistsError("one-use raw, journal, or output path already exists")
    contract = _mapping(
        json.loads(CONTRACT_PATH.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract)
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=False, exist_ok=False)
    probe = RAW_DIR / ".write-probe"
    write_bytes_atomic(probe, b"ready\n")
    probe.unlink()
    journal: dict[str, object] = {
        "schema_version": "binance-tsm-bstock-dividend-underdebit-v1-journal",
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_sha256": contract["contract_sha256"],
        "requests": [],
    }
    _write_json(JOURNAL_PATH, journal)
    try:
        issuer_receipts = _capture_sources(contract, journal)
        retained = _mapping(contract["retained_sources"], name="retained sources")
        bstocks = _mapping(
            _load_bound_json(retained["bstock_inventory"]), name="bstock envelope"
        )
        futures = _mapping(
            _load_bound_json(retained["futures_exchange_info"]),
            name="futures exchange info",
        )
        funding = _load_bound_json(retained["tsm_funding_history"])
        if not isinstance(bstocks.get("data"), list):
            raise ValueError("bStock inventory data must be a list")
        if not isinstance(futures.get("symbols"), list):
            raise ValueError("futures symbols must be a list")
        if not isinstance(funding, list):
            raise ValueError("funding history must be a list")

        bstock_rows = [
            _mapping(row, name="bStock row")
            for row in bstocks["data"]
            if _mapping(row, name="bStock row").get("ticker") == "TSM"
        ]
        futures_rows = [
            _mapping(row, name="futures row")
            for row in futures["symbols"]
            if _mapping(row, name="futures row").get("symbol") == "TSMUSDT"
        ]
        if len(bstock_rows) != 1 or len(futures_rows) != 1:
            raise ValueError("exact TSMB/TSMUSDT retained identity is not unique")
        if futures_rows[0].get("status") != "TRADING":
            raise ValueError("TSMUSDT was not retained as trading")

        event = _mapping(contract["historical_event"], name="historical event")
        start_ms = int(event["window_start_ms"])
        end_ms = int(event["window_end_ms"])
        window = [
            _mapping(row, name="funding row")
            for row in funding
            if start_ms
            <= int(_mapping(row, name="funding row")["fundingTime"])
            <= end_ms
        ]
        special = [row for row in window if row.get("rateType") == "Special"]
        if len(special) != 1:
            raise ValueError("historical window must contain exactly one Special row")
        rate = Decimal(str(special[0]["fundingRate"]))
        mark = Decimal(str(special[0]["markPrice"]))
        if rate >= 0:
            raise ValueError("Special funding was not a short debit")
        special_debit = -(rate * mark)
        gross = Decimal(str(event["issuer_gross_dividend_usd_per_adr"]))
        net = Decimal(str(event["issuer_net_dividend_usd_per_adr"]))
        gross_headroom = gross - special_debit
        net_headroom = net - special_debit

        result: dict[str, object] = {
            "schema_version": "binance-tsm-bstock-dividend-underdebit-v1",
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": "reject_first_upcoming_TSMB_dividend_hedge_from_prior_exact_special_funding",
            "contract": {
                "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                "sha256": contract["contract_sha256"],
            },
            "identity": {
                "bstock_symbol": bstock_rows[0]["symbol"],
                "bstock_ticker": bstock_rows[0]["ticker"],
                "bstock_multiplier": str(bstock_rows[0]["multiplier"]),
                "perpetual_symbol": futures_rows[0]["symbol"],
                "perpetual_contract_type": futures_rows[0]["contractType"],
                "perpetual_status": futures_rows[0]["status"],
            },
            "historical_event": {
                "ex_dividend_date": event["ex_dividend_date"],
                "issuer_gross_dividend_usd_per_adr": format(gross, "f"),
                "issuer_net_dividend_usd_per_adr": format(net, "f"),
                "funding_window_row_count": len(window),
                "special_row_count": len(special),
                "special_funding_time_ms": int(special[0]["fundingTime"]),
                "special_funding_rate": format(rate, "f"),
                "special_mark_price_usdt": format(mark, "f"),
                "matched_short_special_debit_usdt": format(special_debit, "f"),
                "gross_dividend_minus_special_debit": format(gross_headroom, "f"),
                "net_dividend_minus_special_debit": format(net_headroom, "f"),
            },
            "upcoming_event": contract["upcoming_event"],
            "adjudication": {
                "status": "rejected_before_books_prior_TSM_special_funding_exceeded_gross_and_net_dividend",
                "accepted_edge": False,
                "deployment_ready": False,
                "market_direction_forecast_required": False,
                "profitability_claim": False,
                "public_after_cost_profit_floor_usdt": "0",
                "book_requests_justified": False,
                "next_action": "do_not_capture_TSM_books_or_funding_again_for_the_2026_09_16_event",
            },
            "sources": {
                "issuer_pages": issuer_receipts,
                "retained": retained,
                "funding_api_documentation_url": contract[
                    "funding_api_documentation_url"
                ],
            },
            "authority": contract["authority"],
            "implementation": contract["implementation"],
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        _write_json(OUTPUT_PATH, result)
        journal.update(
            {
                "state": "completed",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "result_sha256": result["result_sha256"],
            }
        )
        _write_json(JOURNAL_PATH, journal)
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
        _write_json(JOURNAL_PATH, journal)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run()
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
