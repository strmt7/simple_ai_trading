"""Source-gate Binance Prediction Trading yield-bearing collateral metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import requests
import yaml

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-prediction-yield-bearing-source-contract-v1-2026-08-30.json"
)
OUTPUT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-prediction-yield-bearing-candidate-v1-2026-08-30.json"
)
RAW_DIR = ROOT / (
    "docs/model-research/binance/raw/"
    "prediction-yield-bearing-schema-v1-2026-08-30"
)
JOURNAL_PATH = RAW_DIR / "request-journal.json"
RAW_PATH = RAW_DIR / "prediction-trading-schema.raw.yaml"


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


def _load_contract() -> dict[str, Any]:
    contract = _mapping(
        json.loads(CONTRACT_PATH.read_text(encoding="ascii")), name="contract"
    )
    if contract.get("schema_version") != (
        "binance-prediction-yield-bearing-source-contract-v1"
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
    retained = _mapping(contract["retained_source"], name="retained source")
    retained_path = ROOT / str(retained["path"])
    if _sha256(retained_path.read_bytes()) != retained["sha256"]:
        raise ValueError("retained source hash mismatch")
    return contract


def _operation(schema: Mapping[str, object], endpoint: str) -> dict[str, Any]:
    paths = _mapping(schema.get("paths"), name="paths")
    path_item = _mapping(paths.get(endpoint), name=f"path {endpoint}")
    return _mapping(path_item.get("get"), name=f"GET {endpoint}")


def _parameter_names(operation: Mapping[str, object]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for value in operation.get("parameters", []):
        parameter = _mapping(value, name="parameter")
        name = str(parameter.get("name"))
        result[name] = parameter.get("required") is True
    return result


def _security_names(operation: Mapping[str, object]) -> list[str]:
    names: set[str] = set()
    for value in operation.get("security", []):
        security = _mapping(value, name="security requirement")
        names.update(str(name) for name in security)
    return sorted(names)


def _field_occurrences(value: object, target: str) -> int:
    if isinstance(value, Mapping):
        return sum(
            (1 if str(key) == target else 0)
            + _field_occurrences(child, target)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(_field_occurrences(child, target) for child in value)
    return 0


def _preflight() -> dict[str, Any]:
    contract = _load_contract()
    if RAW_DIR.exists() or OUTPUT_PATH.exists():
        raise FileExistsError("one-use raw directory or output already exists")
    parent = RAW_DIR.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"raw parent missing: {parent}")
    return contract


def run() -> dict[str, object]:
    contract = _preflight()
    RAW_DIR.mkdir(parents=False, exist_ok=False)
    probe = RAW_DIR / ".write-probe"
    write_bytes_atomic(probe, b"ready\n")
    probe.unlink()
    source = _mapping(contract["public_source"], name="public source")
    request: dict[str, object] = {
        "method": "GET",
        "url": source["url"],
        "request_body_sha256": _sha256(b""),
        "state": "planned",
        "planned_at_ms": time.time_ns() // 1_000_000,
    }
    journal: dict[str, object] = {
        "schema_version": "binance-prediction-yield-bearing-source-journal-v1",
        "contract_sha256": contract["contract_sha256"],
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "requests": [request],
    }
    _write_json(JOURNAL_PATH, journal)
    try:
        response = requests.get(
            str(source["url"]),
            headers={
                "Accept": "application/yaml, text/yaml, application/octet-stream",
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            },
            timeout=30,
        )
        write_bytes_atomic(RAW_PATH, response.content)
        request.update(
            {
                "state": "received",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "status_code": response.status_code,
                "response_bytes": len(response.content),
                "response_sha256": _sha256(response.content),
                "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
            }
        )
        _write_json(JOURNAL_PATH, journal)
        response.raise_for_status()
        if len(response.content) > int(source["maximum_response_bytes"]):
            raise ValueError("response exceeds frozen byte ceiling")
        text = response.content.decode("utf-8")
        missing = [
            phrase
            for phrase in source["required_text"]
            if str(phrase) not in text
        ]
        if missing:
            raise ValueError(f"required schema text missing: {missing}")
        schema = _mapping(yaml.safe_load(text), name="OpenAPI schema")
        endpoints = list(contract["scoped_endpoints"])
        operations = {endpoint: _operation(schema, endpoint) for endpoint in endpoints}
        endpoint_contract = {
            endpoint: {
                "security_names": _security_names(operation),
                "parameters_required": _parameter_names(operation),
            }
            for endpoint, operation in operations.items()
        }
        field_occurrences = {
            field: _field_occurrences(schema, field)
            for field in contract["required_schema_fields"]
        }
        source_gate_passed = all(field_occurrences.values()) and all(
            details["parameters_required"].get("timestamp") is True
            and bool(details["security_names"])
            for details in endpoint_contract.values()
        )
        result: dict[str, object] = {
            "schema_version": "binance-prediction-yield-bearing-candidate-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "purpose": (
                "adjudicate_whether_current_Binance_Prediction_Trading_schema_"
                "exposes_a_distinct_direction_independent_yield_bearing_"
                "collateral_candidate_without_accessing_markets_or_accounts"
            ),
            "authority": {
                "public_unauthenticated_documentation_requests": 1,
                "prediction_market_requests": 0,
                "credentials_used": False,
                "account_requests": 0,
                "orders_trades_transfers_or_redemptions": 0,
                "protected_polymarket_capture_accessed": False,
            },
            "source_contract": {
                "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                "sha256": contract["contract_sha256"],
            },
            "retained_schema": {
                "path": RAW_PATH.relative_to(ROOT).as_posix(),
                "sha256": _sha256(response.content),
                "bytes": len(response.content),
                "endpoint_contract": endpoint_contract,
                "field_occurrences": field_occurrences,
                "source_gate_passed": source_gate_passed,
            },
            "mechanism": {
                "market_level_metadata": [
                    "vendor",
                    "chainId",
                    "collateral",
                    "feeRateBps",
                    "slippageBps",
                    "isYieldBearing",
                ],
                "direction_independent_candidate_identity": (
                    "exact_realized_yield_credited_to_independently_justified_"
                    "prediction_collateral_minus_every_incremental_fee_basis_"
                    "conversion_settlement_custody_tax_and_operating_cost"
                ),
                "public_forward_profit_floor": "0",
            },
            "adjudication": {
                "status": (
                    "distinct_source_bound_prediction_collateral_carry_"
                    "candidate_blocked_before_signed_market_access"
                ),
                "accepted_edge": False,
                "accepted_edge_count_change": 0,
                "market_direction_forecast_required": False,
                "profitability_claim": False,
                "deployment_ready": False,
                "reason": (
                    "the_schema_exposes_a_per_market_isYieldBearing_boolean_"
                    "but_does_not_define_yield_recipient_rate_base_accrual_"
                    "distribution_redemption_or_after_cost_economics_and_both_"
                    "market_list_and_detail_are_signed"
                ),
                "retry_trigger": (
                    "both_designated_credentials_plus_explicit_signed_GET_only_"
                    "Prediction_Trading_market_metadata_authority_or_a_material_"
                    "current_official_terms_change_that_defines_yield_"
                    "recipient_rate_base_accrual_distribution_and_redemption"
                ),
            },
            "prohibited": [
                "treating_isYieldBearing_true_as_user_owned_or_positive_yield",
                "using_schema_example_values_as_current_market_economics",
                "calling_any_signed_prediction_endpoint_without_explicit_authority",
                "funding_trading_transferring_redeeming_or_reconfiguring_an_account",
                "double_counting_Polymarket_holding_yield_or_any_vendor_level_yield",
            ],
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        _write_json(OUTPUT_PATH, result)
        journal["state"] = "completed"
        journal["completed_at_ms"] = time.time_ns() // 1_000_000
        journal["result_sha256"] = result["result_sha256"]
        _write_json(JOURNAL_PATH, journal)
        return result
    except Exception as exc:
        journal["state"] = "failed"
        journal["completed_at_ms"] = time.time_ns() // 1_000_000
        journal["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(JOURNAL_PATH, journal)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        contract = _preflight()
        print(
            _canonical_json(
                {
                    "status": "preflight_passed",
                    "contract_sha256": contract["contract_sha256"],
                }
            )
        )
        return
    print(_canonical_json(run()))


if __name__ == "__main__":
    main()
