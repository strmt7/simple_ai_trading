"""Source-gate Binance RPI maker-first hedge economics before market data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    "binance-rpi-maker-hedge-source-contract-v1-2026-08-30.json"
)
OUTPUT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-rpi-maker-hedge-source-gate-v1-2026-08-30.json"
)
RAW_DIR = ROOT / "data/binance-rpi-maker-hedge-source-gate-v1/raw"
JOURNAL_PATH = ROOT / "data/binance-rpi-maker-hedge-source-gate-v1/journal.json"


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


def _validate_contract(contract: Mapping[str, object]) -> None:
    if contract.get("schema_version") != "binance-rpi-maker-hedge-source-v1-contract":
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
    retained = _mapping(contract["retained_sources"], name="retained sources")
    for source in retained.values():
        bound = _mapping(source, name="retained source")
        payload = (ROOT / str(bound["path"])).read_bytes()
        if _sha256(payload) != bound["sha256"]:
            raise ValueError(f"retained source hash mismatch: {bound['path']}")


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
    for request_id, raw_source in enumerate(contract["public_sources"]):
        source = _mapping(raw_source, name="public source")
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
        text = response.content.decode("utf-8").lower()
        missing = [
            value
            for value in source["required_text_casefolded"]
            if str(value).lower() not in text
        ]
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
        "schema_version": "binance-rpi-maker-hedge-source-v1-journal",
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_sha256": contract["contract_sha256"],
        "requests": [],
    }
    _write_json(JOURNAL_PATH, journal)
    try:
        source_receipts = _capture_sources(contract, journal)
        result: dict[str, object] = {
            "schema_version": "binance-rpi-maker-hedge-source-gate-v1",
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": "source_gate_RPI_maker_first_hedge_before_market_data",
            "contract": {
                "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                "sha256": contract["contract_sha256"],
            },
            "mechanism": {
                "rpi_order_type": "post_only_retail_price_improvement_liquidity",
                "ordinary_depth_excludes_rpi": True,
                "rpi_depth_includes_and_aggregates_rpi": True,
                "crossed_rpi_levels_hidden": True,
                "recent_trade_rows_identify_rpi_fills": True,
                "account_specific_rpi_commission_exists": True,
                "standalone_fill_is_directional_inventory": True,
                "direction_independent_form": (
                    "one_owned_RPI_maker_fill_followed_by_an_immediate_equal_base_"
                    "opposite_hedge_with_complete_orphan_and_funding_costs"
                ),
            },
            "adjudication": {
                "status": "candidate_blocked_before_books_by_unknown_account_RPI_commission_and_fill_path",
                "accepted_edge": False,
                "deployment_ready": False,
                "market_direction_forecast_required": False,
                "profitability_claim": False,
                "public_after_cost_profit_floor_bips": "0",
                "public_rpi_depth_requests_justified_now": False,
                "reason": (
                    "the public book cannot establish after_cost_headroom_without_"
                    "the exact account_RPI_commission_and_an_owned_fill_cannot_be_"
                    "assumed_from_visible_depth"
                ),
                "retry_trigger": (
                    "both_designated_Binance_credentials_plus_explicit_signed_GET_"
                    "only_authority_for_one_exact_symbol_RPI_commission_query_and_"
                    "an_independently_positive_organic_equal_base_hedge_question;_"
                    "every_RPI_order_or_hedge_requires_separate_trade_authority"
                ),
            },
            "sources": {
                "public_pages": source_receipts,
                "retained": contract["retained_sources"],
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
