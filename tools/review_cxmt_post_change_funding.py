"""One frozen CXMT post-change funding-only study, with no price/account access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

from tools.capture_public_source_bounded import capture

HOUR = 3_600_000
FIRST = int(datetime(2026, 9, 4, 12, tzinfo=UTC).timestamp() * 1000)
ENDS = tuple(FIRST + i * 4 * HOUR for i in range(12))
HOURS = tuple(FIRST - 3 * HOUR + i * HOUR for i in range(48))
ORIENTATION = "long_polymarket_short_binance"


def _hash(value: dict, field: str) -> str:
    clean = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            clean,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _rate(value: object) -> Decimal:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("funding rate must be a bounded decimal string")
    try:
        rate = Decimal(value)
    except InvalidOperation:
        raise ValueError("funding rate has invalid decimal syntax") from None
    if not rate.is_finite() or abs(rate) > 1:
        raise ValueError("funding rate is not a finite fractional rate")
    return rate


def _hour(value: object) -> int:
    if type(value) is not int:
        raise ValueError("funding timestamp must be integer UTC milliseconds")
    nominal = ((value + HOUR // 2) // HOUR) * HOUR
    if abs(value - nominal) > 60_000:
        raise ValueError("funding timestamp exceeds frozen one-minute phase tolerance")
    return nominal


def evaluate(polymarket: object, binance: object) -> dict:
    """Require exact 48-hour coverage before evaluating the fixed 6/3/3 roles."""
    if not isinstance(polymarket, dict) or polymarket.get("more") is not False:
        raise ValueError("Polymarket page is incomplete")
    poly_rows = polymarket.get("data")
    if not isinstance(poly_rows, list) or len(poly_rows) != 48:
        raise ValueError("Polymarket must contain exactly 48 hourly settlements")
    poly: dict[int, Decimal] = {}
    for row in poly_rows:
        if not isinstance(row, dict):
            raise ValueError("invalid Polymarket funding row")
        timestamp = _hour(row.get("timestamp"))
        if timestamp in poly:
            raise ValueError("duplicate Polymarket nominal hour")
        poly[timestamp] = _rate(row.get("funding_rate"))
    if set(poly) != set(HOURS):
        raise ValueError("Polymarket settlement population differs from frozen hours")
    if not isinstance(binance, list) or len(binance) != 12:
        raise ValueError("Binance must contain exactly twelve regular settlements")
    binance_rates: dict[int, Decimal] = {}
    for row in binance:
        if not isinstance(row, dict) or row.get("symbol") != "CXMTUSDT":
            raise ValueError("Binance symbol identity mismatch")
        if row.get("rateType") != "Regular":
            raise ValueError(
                "nonregular funding requires separate matched cash-flow qualification"
            )
        timestamp = _hour(row.get("fundingTime"))
        if timestamp in binance_rates:
            raise ValueError("duplicate Binance nominal settlement")
        rate = _rate(row.get("fundingRate"))
        if abs(rate) > Decimal("0.01"):
            raise ValueError("regular funding exceeds source-bound post-change cap")
        binance_rates[timestamp] = rate
    if set(binance_rates) != set(ENDS):
        raise ValueError("Binance settlement population differs from frozen twelve")
    with localcontext() as context:
        context.prec = 60
        rows = []
        for timestamp in ENDS:
            poly_rate = sum((poly[timestamp - i * HOUR] for i in range(4)), Decimal(0))
            carry = (binance_rates[timestamp] - poly_rate) * 10_000
            rows.append(
                {
                    "end_ms": timestamp,
                    "binance_rate": str(binance_rates[timestamp]),
                    "polymarket_four_hour_rate": str(poly_rate),
                    "gross_equal_notional_carry_bips": str(carry),
                }
            )
        roles = {}
        for name, selected in (
            ("training", rows[:6]),
            ("validation", rows[6:9]),
            ("test", rows[9:]),
        ):
            gross = sum(
                (Decimal(row["gross_equal_notional_carry_bips"]) for row in selected),
                Decimal(0),
            )
            capital = Decimal(1000) * (len(selected) * 4) / 8760
            net = gross - 20 - capital
            stressed = net - 10
            positive = sum(
                Decimal(row["gross_equal_notional_carry_bips"]) > 0 for row in selected
            )
            worst_drop = min(
                gross
                - Decimal(row["gross_equal_notional_carry_bips"])
                - 20
                - capital
                - 10
                for row in selected
            )
            roles[name] = {
                "count": len(selected),
                "gross_bips": str(gross),
                "capital_hurdle_bips": str(capital),
                "net_after_execution_and_capital_bips": str(net),
                "net_after_quote_stress_bips": str(stressed),
                "positive_count": positive,
                "worst_drop_one_with_full_costs_bips": str(worst_drop),
                "passes": stressed > 0
                and positive * 4 >= len(selected) * 3
                and worst_drop > 0,
            }
    return {
        "rows": rows,
        "roles": roles,
        "fixed_orientation": ORIENTATION,
        "history_survivor": all(role["passes"] for role in roles.values()),
        "qualified_edge": False,
        "notional_mark_path_qualified": False,
        "cross_regime_qualified": False,
    }


def run(plan_path: Path, *, preflight: bool = False) -> dict | None:
    plan = json.loads(plan_path.read_bytes())
    if plan.get("plan_sha256") != _hash(plan, "plan_sha256"):
        raise ValueError("study plan hash mismatch")
    if (
        plan["orientation"] != ORIENTATION
        or plan["expected_binance_end_ms"] != list(ENDS)
        or plan["expected_polymarket_hour_ms"] != list(HOURS)
    ):
        raise ValueError("study population or orientation changed")
    if datetime.now(UTC) < datetime(2026, 9, 6, 8, 10, tzinfo=UTC):
        raise ValueError("post-change settlement gate is not due")
    for path, digest in plan["source_bindings"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("study implementation or trigger binding changed")
    output = Path(plan["result_path"])
    if output.exists() or not output.parent.is_dir():
        raise ValueError("study result exists or its parent is missing")
    contracts = [Path(path) for path in plan["capture_contracts"]]
    if len(contracts) != 2:
        raise ValueError("exactly two frozen source contracts required")
    for contract in contracts:
        capture(contract, preflight=True)
    if preflight:
        return None
    source_results, raw_paths = [], []
    result = {
        "schema_version": 1,
        "plan_path": plan_path.as_posix(),
        "plan_sha256": plan["plan_sha256"],
        "source_results": source_results,
        "accepted_edge": False,
        "candidate_for_books": False,
        "status": "source_failed",
    }
    try:
        for contract_path in contracts:
            source = capture(contract_path)
            source_results.append(source)
            if not source["source_gate"]["passed"]:
                raise ValueError("frozen source admission failed")
            raw_paths.append(Path(source["capture"]["receipt"]["raw_path"]))
        result["analysis"] = evaluate(
            *(json.loads(path.read_bytes()) for path in raw_paths)
        )
        result["status"] = (
            "history_survivor_needs_separate_qualification"
            if result["analysis"]["history_survivor"]
            else "terminal_history_rejection"
        )
    except (ValueError, OSError) as failure:
        result["failure"] = {"type": type(failure).__name__, "reason": str(failure)}
    result["completed_at_utc"] = datetime.now(UTC).isoformat()
    result["result_sha256"] = _hash(result, "result_sha256")
    with output.open("x", encoding="ascii", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=True, sort_keys=True, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    outcome = run(arguments.plan, preflight=arguments.preflight)
    print(
        json.dumps(
            {
                "preflight": arguments.preflight,
                "status": None if outcome is None else outcome["status"],
            }
        )
    )
