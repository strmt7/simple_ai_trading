"""Frozen distinct-population option/perpetual rejection screen; no trading."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.adjudicate_binance_crypto_option_distinct_price_prefilter_v2 import (
    SYMBOL_PATTERN,
    UNDERLYINGS,
    _canonical_hash,
    _futures_map,
    _load_source_result,
    _ticker_map,
    _verify_self_hash,
)
from tools.capture_public_source_bounded import capture


ROOT = Path(__file__).resolve().parents[1]


def number(value: object) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("finite nonnegative decimal required")
    return result


def rows_for(metadata: list[dict], tickers: dict, futures: dict) -> list[dict]:
    """Price every frozen unit-one contract using only its acquisition side."""
    rows = []
    for meta in metadata:
        symbol = meta["symbol"]
        match = SYMBOL_PATTERN.fullmatch(symbol)
        if match is None or number(meta["unit"]) != 1:
            raise ValueError("exact supported unit-one symbol required")
        underlying = UNDERLYINGS[match.group("base")]
        if (
            meta.get("underlying") != underlying
            or meta.get("status") != "TRADING"
            or meta.get("quoteAsset") != "USDT"
            or meta.get("contractType") != "CRYPTO_OPTIONS"
            or meta.get("underlyingType") != "CRYPTO"
        ):
            raise ValueError("metadata eligibility differs")
        ticker, future = tickers[symbol], futures[underlying]
        strike = number(ticker["strikePrice"])
        if strike != number(match.group("strike")):
            raise ValueError("strike differs")
        ask = number(ticker["askPrice"])
        call = match.group("side") == "C"
        entry = number(future["bidPrice" if call else "askPrice"])
        eligible = ask > 0 and entry > 0
        gross = entry - strike - ask if call else strike - entry - ask
        stress = entry * Decimal("0.00335")
        rows.append(
            {
                "symbol": symbol,
                "underlying": underlying,
                "option_ask": str(ask),
                "strike": str(strike),
                "perpetual_entry": str(entry),
                "perpetual_side": "bid" if call else "ask",
                "positive_entry_sides": eligible,
                "gross_floor_per_base_usdt": str(gross) if eligible else None,
                "fixed_stress_per_base_usdt": str(stress) if eligible else None,
                "after_fixed_stress_per_base_usdt": str(gross - stress)
                if eligible
                else None,
                "passes_row_gate": eligible and gross > stress,
            }
        )
    return rows


def run(path: Path, *, preflight: bool = False) -> dict | None:
    """Validate all inputs, capture at most two sources, then apply the run-level gate."""
    plan = json.loads(path.read_bytes())
    _verify_self_hash(plan, "contract_sha256", "price plan")
    for source, expected in plan["implementation_sha256"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
            raise ValueError("implementation differs")
    if plan["fixed_stress_bps"] != "33.5" or plan["maximum_start_skew_ms"] != 10000:
        raise ValueError("fixed rejection gates differ")
    population = json.loads((ROOT / plan["population_path"]).read_bytes())
    _verify_self_hash(population, "result_sha256", "population")
    if population["result_sha256"] != plan["population_sha256"]:
        raise ValueError("frozen population differs")
    symbols = population["distinct_symbols"]
    metadata = population["distinct_metadata"]
    if (
        not symbols
        or symbols != sorted(set(symbols))
        or [m["symbol"] for m in metadata] != symbols
    ):
        raise ValueError("complete nonempty sorted population required")
    if any(number(m["unit"]) != 1 for m in metadata):
        raise ValueError("non-unit contract needs separate economic semantics")
    sources = plan["source_results"]
    for binding in sources.values():
        capture(ROOT / binding["contract_path"], preflight=True)
    output = ROOT / plan["output_path"]
    if output.exists():
        raise FileExistsError("price study already consumed")
    if preflight:
        return None
    result = {
        "schema_version": "option-floor-distinct-prefilter-v3",
        "contract_sha256": plan["contract_sha256"],
        "population_sha256": population["result_sha256"],
        "accepted_edge": False,
        "profitability_claim": False,
        "account_requests": 0,
        "all_rows": [],
        "survivors": [],
        "source_results": {},
        "failure_type": None,
    }
    try:
        for name in ("option_tickers", "futures_books"):
            binding = sources[name]
            source = capture(ROOT / binding["contract_path"])
            result["source_results"][name] = source["result_sha256"]
            if not source["source_gate"]["passed"]:
                raise ValueError("source gate failed; stop before further requests")
        options_source, options_raw = _load_source_result(sources["option_tickers"])
        futures_source, futures_raw = _load_source_result(sources["futures_books"])
        receipts = [s["capture"]["receipt"] for s in (options_source, futures_source)]
        skew = abs(receipts[0]["requested_at_ms"] - receipts[1]["requested_at_ms"])
        result["request_start_skew_ms"] = skew
        result["skew_gate_passed"] = skew <= plan["maximum_start_skew_ms"]
        if any(
            m["expiryDate"] <= max(r["completed_at_ms"] for r in receipts)
            for m in metadata
        ):
            raise ValueError("contract expired during capture")
        rows = rows_for(
            metadata,
            _ticker_map(json.loads(options_raw)),
            _futures_map(json.loads(futures_raw)),
        )
        result["all_rows"] = rows
        result["survivors"] = [
            row for row in rows if row["passes_row_gate"] and result["skew_gate_passed"]
        ]
        result["next_action"] = (
            "freeze_full_cost_and_quantity_stress"
            if result["survivors"]
            else "stop_without_depth_or_accounts"
        )
    except Exception as failure:
        result["failure_type"] = type(failure).__name__
        result["next_action"] = "terminal_failure_no_retry"
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    with output.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(args.contract, preflight=args.preflight)
    print(
        json.dumps(
            {
                "preflight": args.preflight,
                "rows": None if result is None else len(result["all_rows"]),
                "survivors": None if result is None else len(result["survivors"]),
                "failure": None if result is None else result["failure_type"],
            }
        )
    )


if __name__ == "__main__":
    main()
