"""Audit two retained Binance Options exchangeInfo crypto populations offline."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "binance-crypto-option-population-delta-retained-contract-v1"
RESULT_SCHEMA_VERSION = "binance-crypto-option-population-delta-retained-result-v1"
UNDERLYINGS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


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


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _load_bound_json(binding: Mapping[str, object]) -> dict[str, object]:
    path = ROOT / str(binding["path"])
    payload = path.read_bytes()
    if _sha256(payload) != binding["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    return _mapping(json.loads(payload), name=str(binding["path"]))


def _selected_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows = payload.get("optionSymbols")
    if not isinstance(rows, list):
        raise ValueError("exchangeInfo optionSymbols must be a list")
    selected = [
        _mapping(value, name="option symbol")
        for value in rows
        if isinstance(value, Mapping)
        and value.get("status") == "TRADING"
        and value.get("contractType") == "CRYPTO_OPTIONS"
        and value.get("underlyingType") == "CRYPTO"
        and value.get("underlying") in UNDERLYINGS
        and value.get("quoteAsset") == "USDT"
        and Decimal(str(value.get("unit"))) == Decimal("1")
    ]
    selected.sort(key=lambda row: str(row["symbol"]))
    symbols = [str(row["symbol"]) for row in selected]
    if len(symbols) != len(set(symbols)):
        raise ValueError("selected option symbols are not unique")
    return selected


def _expiry_groups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter((str(row["underlying"]), int(row["expiryDate"])) for row in rows)
    return [
        {
            "underlying": underlying,
            "expiry_date_ms": expiry,
            "symbol_count": count,
        }
        for (underlying, expiry), count in sorted(counts.items())
    ]


def _validate_contract(contract: Mapping[str, object], *, preflight_only: bool) -> None:
    expected_status = (
        "preflight_only_unconsumed"
        if preflight_only
        else "frozen_before_zero_network_population_delta"
    )
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected contract schema")
    if contract.get("status") != expected_status:
        raise ValueError("contract status does not match invocation mode")
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    implementation = _mapping(contract["implementation"], name="implementation")
    implementation_path = ROOT / str(implementation["path"])
    if _sha256(implementation_path.read_bytes()) != implementation["sha256"]:
        raise ValueError("implementation hash mismatch")
    if contract.get("population_filter") != {
        "allowed_underlyings": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "contract_type": "CRYPTO_OPTIONS",
        "option_unit": "1",
        "quote_asset": "USDT",
        "status": "TRADING",
        "underlying_type": "CRYPTO",
    }:
        raise ValueError("unexpected population filter")
    authority = _mapping(contract["authority"], name="authority")
    if authority != {
        "account_state_accessed": False,
        "authenticated_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "new_public_requests": 0,
        "orders_quotes_transfers_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }:
        raise ValueError("unexpected authority")
    retained = _mapping(contract["retained_sources"], name="retained sources")
    for key, hash_field in (
        ("baseline", "sha256"),
        ("current", "sha256"),
        ("current_capture_receipt", "sha256"),
        ("current_capture_result", "file_sha256"),
    ):
        binding = _mapping(retained[key], name=key)
        path = ROOT / str(binding["path"])
        if _sha256(path.read_bytes()) != binding[hash_field]:
            raise ValueError(f"retained lineage hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    contract_bytes = args.contract.read_bytes()
    contract = _mapping(json.loads(contract_bytes), name="contract")
    _validate_contract(contract, preflight_only=args.preflight_only)
    if args.preflight_only:
        if args.output is not None:
            raise ValueError("preflight must not specify an output")
        print("preflight_passed=true")
        return 0
    if args.output is None:
        raise ValueError("frozen execution requires --output")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite retained delta evidence")

    retained = _mapping(contract["retained_sources"], name="retained sources")
    baseline_binding = _mapping(retained["baseline"], name="baseline source")
    current_binding = _mapping(retained["current"], name="current source")
    baseline_rows = _selected_rows(_load_bound_json(baseline_binding))
    current_rows = _selected_rows(_load_bound_json(current_binding))
    baseline_symbols = [str(row["symbol"]) for row in baseline_rows]
    current_symbols = [str(row["symbol"]) for row in current_rows]
    if len(baseline_symbols) != int(contract["expected_baseline_count"]):
        raise ValueError("baseline eligible option count changed")

    baseline_set = set(baseline_symbols)
    current_set = set(current_symbols)
    new_symbols = sorted(current_set - baseline_set)
    removed_symbols = sorted(baseline_set - current_set)
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": args.contract.as_posix(),
            "sha256": _sha256(contract_bytes),
            "canonical_sha256": contract["contract_sha256"],
        },
        "implementation": contract["implementation"],
        "authority": contract["authority"],
        "retained_sources": retained,
        "population": {
            "baseline_count": len(baseline_symbols),
            "baseline_sorted_symbols_sha256": _sha256(
                "\n".join(baseline_symbols).encode("ascii")
            ),
            "baseline_expiry_groups": _expiry_groups(baseline_rows),
            "current_count": len(current_symbols),
            "current_sorted_symbols_sha256": _sha256(
                "\n".join(current_symbols).encode("ascii")
            ),
            "current_expiry_groups": _expiry_groups(current_rows),
            "new_symbol_count": len(new_symbols),
            "new_symbols": new_symbols,
            "removed_symbol_count": len(removed_symbols),
            "removed_symbols": removed_symbols,
        },
        "adjudication": {
            "literal_rank_47_new_population_trigger_satisfied": bool(new_symbols),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "next_action": (
                "freeze_one_separate_public_price_prefilter_for_only_the_new_symbols"
                if new_symbols
                else "stop_without_ticker_futures_depth_or_funding_requests"
            ),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["population"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
