"""Apply retained activity gates to every exact three-leg conversion candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.storage import write_bytes_atomic
from tools import adjudicate_binance_two_intermediary_conversions_retained as retained
from tools import screen_binance_indirect_internal_conversions as source


CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-activity-contract-v1.json"
)
PARENT_RESULT = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-retained-v3-2026-08-29.json"
)
ACTIVITY_RESULT = Path(
    "docs/model-research/action-value/"
    "binance-indirect-internal-conversion-activity-adjudication-v1-2026-08-29.json"
)
ACTIVITY_RAW = Path(
    "data/edge-research/binance-indirect-internal-conversion-activity-v1/"
    "ticker-24hr.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-activity-adjudication-v1-2026-08-29.json"
)
IMPLEMENTATION_PATH = Path(
    "tools/adjudicate_binance_two_intermediary_conversion_activity.py"
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
    claimed = str(body.pop(field))
    actual = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != actual:
        raise ValueError(f"{field} differs: claimed={claimed} actual={actual}")
    return claimed


def _load_contract() -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if contract.get("status") != "frozen_before_retained_activity_outcome_access":
        raise ValueError("activity contract is not frozen")
    claimed = _canonical_hash(contract, "contract_sha256")
    if (
        _sha256(IMPLEMENTATION_PATH.read_bytes())
        != contract["implementation"]["sha256"]
    ):
        raise ValueError("activity implementation differs from frozen contract")
    return contract, claimed


def _quote_changes(
    symbols: set[str],
    books: Sequence[Mapping[str, Mapping[str, object]]],
) -> dict[str, int]:
    fields = ("bidPrice", "bidQty", "askPrice", "askQty")
    result: dict[str, int] = {}
    for symbol in sorted(symbols):
        previous: tuple[str, ...] | None = None
        changes = 0
        for book in books:
            row = book.get(symbol)
            if row is None:
                previous = None
                continue
            current = tuple(str(row[field]) for field in fields)
            if previous is not None and current != previous:
                changes += 1
            previous = current
        result[symbol] = changes
    return result


def run(output: Path) -> dict[str, object]:
    contract, contract_hash = _load_contract()
    parent = json.loads(PARENT_RESULT.read_text(encoding="ascii"))
    if (
        _canonical_hash(parent, "result_sha256")
        != contract["parent_result"]["result_sha256"]
    ):
        raise ValueError("parent result differs")
    if _sha256(PARENT_RESULT.read_bytes()) != contract["parent_result"]["file_sha256"]:
        raise ValueError("parent result file differs")
    activity = json.loads(ACTIVITY_RESULT.read_text(encoding="ascii"))
    if (
        _canonical_hash(activity, "result_sha256")
        != contract["retained_activity"]["result_sha256"]
    ):
        raise ValueError("retained activity adjudication differs")
    raw_payload = ACTIVITY_RAW.read_bytes()
    if _sha256(raw_payload) != contract["retained_activity"]["raw_sha256"]:
        raise ValueError("retained activity response differs")
    if activity["capture_level_gates"]["all_passed"] is not True:
        raise ValueError("retained activity capture-level gates failed")

    _, raw_books = retained._load_retained_books(contract)
    parsed_books = [source.parse_books(raw) for raw in raw_books]
    candidates = parent["exact_empirical_candidates"]
    symbols = {
        symbol
        for candidate in candidates
        for symbol in [candidate["direct_symbol"], *candidate["three_leg_symbols"]]
    }
    changes = _quote_changes(symbols, parsed_books)
    ticker_rows = json.loads(raw_payload)
    trade_counts = {str(row["symbol"]): int(row["count"]) for row in ticker_rows}
    minimum_changes = int(contract["activity_gates"]["minimum_quote_changes"])
    minimum_trades = int(contract["activity_gates"]["minimum_24h_trade_count"])

    decisions: list[dict[str, object]] = []
    survivors: list[dict[str, object]] = []
    for candidate in candidates:
        required = [candidate["direct_symbol"], *candidate["three_leg_symbols"]]
        symbol_rows = [
            {
                "symbol": symbol,
                "quote_changes": changes.get(symbol, 0),
                "trade_count_24h": trade_counts.get(symbol, 0),
            }
            for symbol in required
        ]
        passed = all(
            row["quote_changes"] >= minimum_changes
            and row["trade_count_24h"] >= minimum_trades
            for row in symbol_rows
        )
        decision = {
            "route_id": candidate["route_id"],
            "start_usdt": candidate["start_usdt"],
            "source": candidate["source"],
            "intermediaries": candidate["intermediaries"],
            "target": candidate["target"],
            "symbols": symbol_rows,
            "minimum_quote_changes": min(row["quote_changes"] for row in symbol_rows),
            "minimum_24h_trade_count": min(
                row["trade_count_24h"] for row in symbol_rows
            ),
            "median_incremental_bips": candidate["median_incremental_bips"],
            "minimum_incremental_bips": candidate["minimum_incremental_bips"],
            "activity_gate_passed": passed,
            "static_route_accepted": False,
        }
        decisions.append(decision)
        if passed:
            survivors.append(decision)
    decisions.sort(key=lambda row: (row["route_id"], row["start_usdt"]))
    survivors.sort(key=lambda row: float(row["median_incremental_bips"]), reverse=True)
    result_without_hash: dict[str, object] = {
        "schema_version": "binance-two-intermediary-conversion-activity-adjudication-v1",
        "status": "completed_zero_network_retained_activity_adjudication",
        "contract_sha256": contract_hash,
        "parent_result_sha256": contract["parent_result"]["result_sha256"],
        "retained_activity_result_sha256": contract["retained_activity"][
            "result_sha256"
        ],
        "population": {
            "exact_parent_candidates": len(candidates),
            "unique_required_symbols": len(symbols),
            "activity_survivors": len(survivors),
            "activity_rejections": len(candidates) - len(survivors),
        },
        "candidate_activity_decisions": decisions,
        "activity_survivors": survivors,
        "decision": {
            "accepted_scoped_mechanism_extension": bool(survivors),
            "accepted_scope": "best-of-direct-one-or-two-intermediary fail-closed cost comparison for an independently required legitimate same-account organic conversion only",
            "static_route_accepted": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_evidence": "exact account symbol commissions and a future independent prospective organic-conversion paper or separately authorized owned sequential completion window",
        },
        "accepted_edge": bool(survivors),
        "deployment_ready": False,
        "authority": {
            "network_requests": 0,
            "credentials_used": False,
            "account_state_accessed": False,
            "orders_or_mutations": 0,
            "protected_capture_accessed": False,
        },
    }
    result = {
        **result_without_hash,
        "result_sha256": _sha256(_canonical_json(result_without_hash).encode("ascii")),
    }
    write_bytes_atomic(output, (_canonical_json(result) + "\n").encode("ascii"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "population": result["population"],
                "accepted_edge": result["accepted_edge"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
