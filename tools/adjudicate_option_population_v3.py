"""Offline complete-union exclusion gate; never reads prices or makes requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.adjudicate_binance_crypto_option_population_gate_v2 import (
    _canonical_hash,
    _eligible_symbols,
    _verify_self_hash,
)


ROOT = Path(__file__).resolve().parents[1]


def known_symbols(plan: dict) -> set[str]:
    """Union every frozen prior population, rather than only the latest delta."""
    known = set()
    for binding in plan["exclusions"]:
        raw = (ROOT / binding["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError("prior population file hash differs")
        value = json.loads(raw)
        if binding["kind"] == "exchange_info":
            symbols = _eligible_symbols(value)
        else:
            _verify_self_hash(value, "result_sha256", binding["path"])
            symbols = value
            for key in binding["keys"]:
                symbols = symbols[key]
        if not isinstance(symbols, list) or not all(
            isinstance(x, str) for x in symbols
        ):
            raise ValueError("prior symbols must be a list of strings")
        known.update(symbols)
    if len(known) != plan["expected_exclusion_count"]:
        raise ValueError("prior union count differs")
    return known


def adjudicate(path: Path, *, preflight: bool = False) -> dict | None:
    """Require frozen source/exclusion bindings before classifying any new symbol."""
    plan = json.loads(path.read_bytes())
    _verify_self_hash(plan, "contract_sha256", "population plan")
    for binding in plan["implementations"]:
        if (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise ValueError("population implementation hash differs")
    known = known_symbols(plan)
    if preflight:
        return None
    output = ROOT / plan["output_path"]
    if output.exists():
        raise FileExistsError("population result already exists")
    source = json.loads((ROOT / plan["source_result_path"]).read_bytes())
    _verify_self_hash(source, "result_sha256", "source result")
    if source["contract"]["sha256"] != plan["source_contract_sha256"]:
        raise ValueError("source contract differs")
    if source["source_gate"]["passed"] is not True:
        raise ValueError("source admission failed; no population or price permission")
    receipt = source["capture"]["receipt"]
    raw = (ROOT / receipt["raw_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["response_sha256"]:
        raise ValueError("current metadata differs")
    metadata = json.loads(raw)
    current = set(_eligible_symbols(metadata))
    distinct = sorted(current - known)
    indexed = {row["symbol"]: row for row in metadata["optionSymbols"]}
    for symbol in distinct:
        expiry = indexed[symbol].get("expiryDate")
        if type(expiry) is not int or expiry <= receipt["completed_at_ms"]:
            raise ValueError("new symbol expiry missing or already passed")
    result = {
        "schema_version": "option-population-union-gate-v3",
        "contract_sha256": plan["contract_sha256"],
        "source_result_sha256": source["result_sha256"],
        "current_eligible_count": len(current),
        "excluded_known_count": len(current & known),
        "distinct_count": len(distinct),
        "distinct_symbols": distinct,
        "distinct_metadata": [indexed[symbol] for symbol in distinct],
        "new_population_trigger_satisfied": bool(distinct),
        "next_action": "freeze_distinct_only_price_prefilter"
        if distinct
        else "stop_without_prices",
        "accepted_edge": False,
        "profitability_claim": False,
        "price_requests": 0,
        "account_requests": 0,
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    with output.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = adjudicate(args.contract, preflight=args.preflight)
    print(
        json.dumps(
            {
                "preflight": args.preflight,
                "distinct_count": None if result is None else result["distinct_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
