"""Source-gate Binance Spot amend-keep-priority using retained production config."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
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
    "binance-spot-amend-keep-priority-source-contract-v1-2026-08-30.json"
)
OUTPUT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-spot-amend-keep-priority-candidate-v1-2026-08-30.json"
)
RAW_DIR = ROOT / "data/binance-spot-amend-keep-priority-source-v1/raw"
JOURNAL_PATH = ROOT / "data/binance-spot-amend-keep-priority-source-v1/journal.json"


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


def _parse_utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("frozen_at_utc lacks an explicit offset")
    return parsed.astimezone(timezone.utc)


def _validate_contract(contract: Mapping[str, object]) -> None:
    if contract.get("schema_version") != (
        "binance-spot-amend-keep-priority-source-v1-contract"
    ):
        raise ValueError("contract schema differs")
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    implementation = _mapping(contract["implementation"], name="implementation")
    if _sha256(Path(__file__).read_bytes()) != implementation["sha256"]:
        raise ValueError("implementation hash mismatch")
    if _parse_utc(contract["frozen_at_utc"]) > datetime.now(timezone.utc):
        raise ValueError("frozen_at_utc is in the future")
    for raw_source in contract["retained_sources"]:
        source = _mapping(raw_source, name="retained source")
        payload = (ROOT / str(source["path"])).read_bytes()
        if _sha256(payload) != source["sha256"]:
            raise ValueError(f"retained source hash mismatch: {source['path']}")


def _load_scoped_configuration(contract: Mapping[str, object]) -> list[dict[str, object]]:
    config = _mapping(contract["retained_exchange_info"], name="exchange info")
    compressed = (ROOT / str(config["path"])).read_bytes()
    decompressed = gzip.decompress(compressed)
    if _sha256(compressed) != config["gzip_sha256"]:
        raise ValueError("exchangeInfo gzip hash mismatch")
    if _sha256(decompressed) != config["payload_sha256"]:
        raise ValueError("exchangeInfo payload hash mismatch")
    payload = _mapping(json.loads(decompressed), name="exchangeInfo payload")
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError("exchangeInfo symbols must be an array")
    selected: list[dict[str, object]] = []
    for symbol in contract["scoped_symbols"]:
        matches = [
            _mapping(row, name="symbol row")
            for row in symbols
            if isinstance(row, Mapping) and row.get("symbol") == symbol
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {symbol} row, found {len(matches)}")
        row = matches[0]
        filters = row.get("filters")
        if not isinstance(filters, list):
            raise ValueError(f"{symbol} filters must be an array")
        amend_filters = [
            _mapping(item, name="amend filter")
            for item in filters
            if isinstance(item, Mapping)
            and item.get("filterType") == "MAX_NUM_ORDER_AMENDS"
        ]
        if len(amend_filters) != 1:
            raise ValueError(f"expected one {symbol} amendment filter")
        selected.append(
            {
                "symbol": symbol,
                "status": row.get("status"),
                "amend_allowed": row.get("amendAllowed"),
                "max_num_order_amends": amend_filters[0].get(
                    "maxNumOrderAmends"
                ),
            }
        )
    return selected


def run() -> dict[str, object]:
    if RAW_DIR.exists() or JOURNAL_PATH.exists() or OUTPUT_PATH.exists():
        raise FileExistsError("one-use raw, journal, or output path already exists")
    contract = _mapping(
        json.loads(CONTRACT_PATH.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract)
    scoped_configuration = _load_scoped_configuration(contract)
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=False, exist_ok=False)
    probe = RAW_DIR / ".write-probe"
    write_bytes_atomic(probe, b"ready\n")
    probe.unlink()
    journal: dict[str, object] = {
        "schema_version": "binance-spot-amend-keep-priority-source-v1-journal",
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_sha256": contract["contract_sha256"],
        "requests": [],
    }
    request = {
        "name": "official-faq",
        "method": "GET",
        "url": contract["public_source"]["url"],
        "request_body_sha256": _sha256(b""),
        "state": "planned",
        "planned_at_ms": time.time_ns() // 1_000_000,
    }
    requests_log = journal["requests"]
    if not isinstance(requests_log, list):
        raise ValueError("journal requests must be an array")
    requests_log.append(request)
    _write_json(JOURNAL_PATH, journal)
    try:
        response = requests.get(
            str(request["url"]),
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            },
            timeout=30,
        )
        raw_path = RAW_DIR / "order_amend_keep_priority.raw.md"
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
        source = _mapping(contract["public_source"], name="public source")
        text = response.content.decode("utf-8").casefold()
        missing = [
            phrase
            for phrase in source["required_text_casefolded"]
            if str(phrase).casefold() not in text
        ]
        if missing:
            raise ValueError(f"required FAQ text missing: {missing}")
        configuration_passed = all(
            row["status"] == "TRADING"
            and row["amend_allowed"] is True
            and row["max_num_order_amends"] == 10
            for row in scoped_configuration
        )
        result: dict[str, object] = {
            "schema_version": "binance-spot-amend-keep-priority-candidate-v1",
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": (
                "test_a_direction_independent_queue_priority_overlay_for_an_"
                "independently_required_Spot_maker_order_quantity_reduction"
            ),
            "contract": {
                "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                "sha256": contract["contract_sha256"],
            },
            "mechanism": {
                "allowed_change": "reduce_existing_order_quantity_only",
                "amended_order_id_stays_same": True,
                "amended_order_keeps_same_price_time_priority": True,
                "cancel_replace_loses_time_priority": True,
                "failed_amend_leaves_order_unchanged": True,
                "unfilled_order_count": 0,
                "request_weight": 4,
                "scoped_production_configuration": scoped_configuration,
                "configuration_passed": configuration_passed,
            },
            "economics": {
                "direction_independent_form": (
                    "for_the_same_existing_order_price_and_same_reduced_quantity_"
                    "successful_in_place_amend_weakly_dominates_cancel_replace_"
                    "on_queue_position"
                ),
                "public_forward_profit_floor_quote_units": "0",
                "known_direct_trading_fee_increment": "0",
                "economic_value_identity": (
                    "additional_same_price_quantity_filled_before_cancel_replace_"
                    "would_reach_the_front_times_owned_after_cost_value_per_fill"
                ),
                "value_quantified": False,
            },
            "adjudication": {
                "status": (
                    "material_direction_independent_execution_candidate_blocked_"
                    "before_any_order_probe"
                ),
                "accepted_edge": False,
                "accepted_edge_count_change": 0,
                "deployment_ready": False,
                "market_direction_forecast_required": False,
                "profitability_claim": False,
                "reason": (
                    "the_queue_priority_dominance_is_source_bound_and_enabled_for_"
                    "BTCUSDT_ETHUSDT_SOLUSDT_but_public_evidence_has_no_"
                    "independently_required_owned_order_queue_counterfactual_fill_"
                    "adverse_selection_latency_or_after_cost_value"
                ),
                "next_trigger": (
                    "an_independently_required_existing_BTCUSDT_ETHUSDT_or_"
                    "SOLUSDT_maker_order_quantity_reduction_plus_explicit_"
                    "testnet_or_paper_order_authority_and_exact_owned_queue_fill_"
                    "reconciliation_or_a_material_official_semantics_weight_"
                    "filter_or_configuration_change"
                ),
            },
            "sources": {
                "public_faq": {
                    key: request[key]
                    for key in (
                        "method",
                        "url",
                        "status_code",
                        "response_bytes",
                        "response_sha256",
                        "raw_path",
                    )
                },
                "retained": contract["retained_sources"],
                "retained_exchange_info": contract["retained_exchange_info"],
                "journal": JOURNAL_PATH.relative_to(ROOT).as_posix(),
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
