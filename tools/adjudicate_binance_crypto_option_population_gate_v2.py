"""Adjudicate the frozen Binance crypto-option population gate offline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "binance-crypto-option-population-gate-contract-v2"
RESULT_SCHEMA = "binance-crypto-option-population-gate-result-v2"
ALLOWED_UNDERLYINGS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
PRIOR_POPULATION_PATH = (
    "docs/model-research/action-value/"
    "binance-crypto-option-population-delta-retained-result-v1-2026-08-31.json"
)
PRIOR_PRICE_PATH = (
    "docs/model-research/action-value/"
    "binance-crypto-option-population-price-prefilter-result-v1-2026-08-31.json"
)
LATE_DELTA_PATH = (
    "docs/model-research/action-value/"
    "binance-crypto-option-late-ticker-delta-result-v1-2026-09-01.json"
)


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


def _canonical_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _verify_self_hash(value: Mapping[str, object], field: str, name: str) -> None:
    if _canonical_hash(value, field) != value.get(field):
        raise ValueError(f"{name} canonical hash mismatch")


def _eligible_symbols(value: Mapping[str, object]) -> list[str]:
    rows = value.get("optionSymbols")
    if not isinstance(rows, list):
        raise ValueError("optionSymbols must be a list")
    symbols: list[str] = []
    for value_row in rows:
        if not isinstance(value_row, Mapping):
            continue
        row = dict(value_row)
        try:
            unit = Decimal(str(row.get("unit", "0")))
        except Exception as exc:
            raise ValueError("option unit is not decimal") from exc
        if (
            row.get("underlying") in ALLOWED_UNDERLYINGS
            and row.get("status") == "TRADING"
            and row.get("contractType") == "CRYPTO_OPTIONS"
            and row.get("underlyingType") == "CRYPTO"
            and row.get("quoteAsset") == "USDT"
            and unit > 0
        ):
            symbols.append(str(row["symbol"]))
    if len(symbols) != len(set(symbols)):
        raise ValueError("eligible option symbols are not unique")
    return sorted(symbols)


def _sorted_symbols_sha256(symbols: list[str]) -> str:
    return _sha256(("\n".join(symbols) + "\n").encode("ascii"))


def _load_result(path_text: str) -> dict[str, Any]:
    value = _load_object(ROOT / path_text)
    _verify_self_hash(value, "result_sha256", path_text)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("refusing to overwrite population-gate evidence")
    contract_path = args.contract.resolve()
    source_result_path = args.source_result.resolve()
    contract = _load_object(contract_path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unexpected contract schema")
    _verify_self_hash(contract, "contract_sha256", "population contract")

    source = _load_object(source_result_path)
    _verify_self_hash(source, "result_sha256", "source result")
    source_contract = contract["source_contract"]
    if not isinstance(source_contract, Mapping):
        raise ValueError("source_contract must be an object")
    if source.get("contract") != source_contract:
        raise ValueError("source result contract binding mismatch")
    if source.get("source_gate", {}).get("passed") is not True:
        raise ValueError("source gate did not pass")
    receipt = source.get("capture", {}).get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("source receipt is missing")
    current_path = ROOT / str(receipt["raw_path"])
    current_bytes = current_path.read_bytes()
    if _sha256(current_bytes) != receipt.get("response_sha256"):
        raise ValueError("current exchangeInfo hash mismatch")

    baseline_binding = contract["baseline"]
    if not isinstance(baseline_binding, Mapping):
        raise ValueError("baseline must be an object")
    baseline_path = ROOT / str(baseline_binding["path"])
    baseline_bytes = baseline_path.read_bytes()
    if _sha256(baseline_bytes) != baseline_binding.get("sha256"):
        raise ValueError("baseline exchangeInfo hash mismatch")

    baseline_symbols = _eligible_symbols(json.loads(baseline_bytes))
    current_symbols = _eligible_symbols(json.loads(current_bytes))
    if len(baseline_symbols) != baseline_binding.get("expected_eligible_symbol_count"):
        raise ValueError("baseline eligible count mismatch")
    baseline_set = set(baseline_symbols)
    current_set = set(current_symbols)
    new_symbols = sorted(current_set - baseline_set)
    removed_symbols = sorted(baseline_set - current_set)

    prior_population = _load_result(PRIOR_POPULATION_PATH)
    prior_price = _load_result(PRIOR_PRICE_PATH)
    late_delta = _load_result(LATE_DELTA_PATH)
    prior_symbols = set(prior_population["population"]["new_symbols"])
    late_symbols = set(late_delta["population"]["new_symbols"])
    new_set = set(new_symbols)
    prior_overlap = sorted(new_set & prior_symbols)
    late_overlap = sorted(new_set & late_symbols)
    already_screened = prior_symbols | late_symbols
    unscreened_symbols = sorted(new_set - already_screened)

    if prior_price["population"]["new_symbol_count"] != len(prior_symbols):
        raise ValueError("prior price screen did not cover the prior population")
    if prior_price["population"]["after_fixed_stress_positive_count"] != 0:
        raise ValueError("prior price screen unexpectedly had a survivor")

    underlying_counts = {
        underlying: sum(
            symbol.startswith(underlying.removesuffix("USDT") + "-")
            for symbol in unscreened_symbols
        )
        for underlying in sorted(ALLOWED_UNDERLYINGS)
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": contract["contract_sha256"],
        },
        "source_binding": {
            "source_result_path": str(source_result_path.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "source_result_sha256": source["result_sha256"],
            "raw_path": str(current_path.relative_to(ROOT)).replace("\\", "/"),
            "raw_sha256": receipt["response_sha256"],
            "journal_path": "data/binance-crypto-option-population-gate-v2-2026-09-04/request-journal.jsonl",
            "journal_sha256": _sha256(
                (
                    ROOT
                    / "data/binance-crypto-option-population-gate-v2-2026-09-04/request-journal.jsonl"
                ).read_bytes()
            ),
            "baseline_path": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
            "baseline_sha256": baseline_binding["sha256"],
        },
        "population": {
            "baseline_eligible_symbol_count": len(baseline_symbols),
            "current_eligible_symbol_count": len(current_symbols),
            "new_symbol_count": len(new_symbols),
            "removed_symbol_count": len(removed_symbols),
            "new_symbols_sha256": _sorted_symbols_sha256(new_symbols),
            "new_symbols": new_symbols,
            "previous_508_overlap_count": len(prior_overlap),
            "previous_508_overlap_symbols": prior_overlap,
            "late_delta_overlap_count": len(late_overlap),
            "late_delta_overlap_symbols": late_overlap,
            "distinct_unscreened_symbol_count": len(unscreened_symbols),
            "distinct_unscreened_symbols_sha256": _sorted_symbols_sha256(
                unscreened_symbols
            ),
            "distinct_unscreened_underlying_counts": underlying_counts,
            "distinct_unscreened_symbols": unscreened_symbols,
        },
        "prior_consumed_bindings": {
            "population_508": {
                "path": PRIOR_POPULATION_PATH,
                "result_sha256": prior_population["result_sha256"],
            },
            "price_screen_508": {
                "path": PRIOR_PRICE_PATH,
                "result_sha256": prior_price["result_sha256"],
            },
            "late_delta_2": {
                "path": LATE_DELTA_PATH,
                "result_sha256": late_delta["result_sha256"],
            },
        },
        "preflight_correction": {
            "network_requests_before_correction": 0,
            "outputs_before_correction": 0,
            "validation_failure": "frozen_at_utc_was_later_than_the_actual_clock",
            "correction": "replaced_only_frozen_at_utc_with_the_observed_2026-09-04T18:40:01Z_and_recomputed_self_hashes_before_access",
            "population_rule_changed": False,
            "decision_rules_changed": False,
            "request_changed": False,
        },
        "adjudication": {
            "literal_rank_47_new_population_trigger_satisfied": bool(new_symbols),
            "distinct_unscreened_population_exists": bool(unscreened_symbols),
            "next_action": (
                "freeze_one_separate_two_request_option_ticker_and_futures_book_prefilter_for_only_the_354_distinct_unscreened_symbols"
                if unscreened_symbols
                else "terminalize_without_price_or_depth_access"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": {
            "account_state_accessed": False,
            "authenticated_requests": 0,
            "credentials_used": False,
            "funds_used": False,
            "public_unauthenticated_GET_requests": 1,
            "orders_quotes_transfers_or_wallet_actions": 0,
            "paper_or_live_trading_authority": False,
            "protected_capture_touched": False,
        },
        "implementation": {
            "path": "tools/adjudicate_binance_crypto_option_population_gate_v2.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output_bytes = (_canonical_json(result) + "\n").encode("ascii")
    write_bytes_atomic(args.output, output_bytes)
    print(
        _canonical_json(
            {
                "current": len(current_symbols),
                "new": len(new_symbols),
                "distinct_unscreened": len(unscreened_symbols),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
