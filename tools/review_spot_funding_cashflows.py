"""Offline fixed-base funding accounting; not a rerun of old acceptance gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal as D
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04"
RAW_BASE = ROOT / "data/binance-broad-crypto-funding-carry-preflight-v1/raw"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: dict, field: str) -> str:
    return digest(
        json.dumps(
            {k: v for k, v in value.items() if k != field},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def decimal(value, *, positive=False) -> D:
    if isinstance(value, bool):
        raise ValueError("boolean is not a financial value")
    parsed = D(str(value))
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError("finite financial value required; positive where specified")
    return parsed


def normalize_history(raw: list, symbol: str) -> list[dict]:
    rows = []
    for row in raw:
        time = row["fundingTime"]
        if isinstance(time, bool) or not isinstance(time, int) or time <= 0:
            raise ValueError("positive integer funding timestamp required")
        if row["symbol"] != symbol:
            raise ValueError("funding symbol mismatch")
        rows.append(
            {
                "time": time,
                "rate": decimal(row["fundingRate"]),
                "mark": decimal(row["markPrice"], positive=True),
            }
        )
    rows.sort(key=lambda row: row["time"])
    if not rows or len({row["time"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique funding history required")
    return rows


def cashflow_metrics(
    rows: list[dict], reference: D, days: D, fee: D, legs: D, annual: D
) -> dict[str, str]:
    reference, days, legs = (decimal(x, positive=True) for x in (reference, days, legs))
    fee, annual = decimal(fee), decimal(annual)
    if fee < 0 or annual < 0 or not rows:
        raise ValueError("nonnegative costs and nonempty rows required")
    rates = [decimal(row["rate"]) for row in rows]
    marks = [decimal(row["mark"], positive=True) for row in rows]
    cash = sum((rate * mark for rate, mark in zip(rates, marks, strict=True)), D(0))
    weighted = cash / reference * 10_000
    unweighted = sum(rates, D(0)) * 10_000
    after_fee = weighted - fee
    capital = legs * annual * days / 365
    values = {
        "reference_mark_quote_per_base": reference,
        "funding_cash_quote_per_base": cash,
        "fixed_base_funding_bps_at_reference": weighted,
        "original_unit_weight_funding_bps": unweighted,
        "weighting_difference_bps": weighted - unweighted,
        "after_execution_sensitivity_before_capital_bps": after_fee,
        "original_capital_stress_bps": capital,
        "after_execution_and_original_capital_stress_bps": after_fee - capital,
        "break_even_annual_capital_cost_bps_per_reference_leg": after_fee
        * 365
        / days
        / legs,
    }
    return {key: str(value) for key, value in values.items()}


def run(output: Path) -> None:
    journal_path = output.with_suffix(".journal.jsonl")
    if output.exists() or journal_path.exists():
        raise RuntimeError("offline audit output or journal already exists")
    bindings = []

    def read(path: Path, expected: str | None = None) -> bytes:
        path = path.resolve()
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        checksum = digest(raw)
        if expected is not None and checksum != expected:
            raise ValueError(f"retained bytes differ: {relative}")
        bindings.append({"path": relative, "sha256": checksum})
        return raw

    with journal_path.open("x", encoding="utf-8", newline="\n") as journal:

        def record(phase: str) -> None:
            journal.write(
                json.dumps(
                    {"phase": phase, "time_utc": datetime.now(timezone.utc).isoformat()}
                )
                + "\n"
            )
            journal.flush()
            os.fsync(journal.fileno())

        record("started_offline_accounting_audit")
        try:
            plan = json.loads(read(BASE / "spot-funding-cashflow-plan.json"))
            prior = json.loads(read(ROOT / plan["input_result"]["path"]))
            contract = json.loads(read(ROOT / plan["input_contract"]["path"]))
            for value, source, field in (
                (prior, plan["input_result"], "result_sha256"),
                (contract, plan["input_contract"], "contract_sha256"),
            ):
                if (
                    canonical(value, field) != source[field]
                    or value[field] != source[field]
                ):
                    raise ValueError("original self-hash mismatch")
            gates = contract["economic_and_stability_gates"]
            if (
                gates["round_trip_execution_stress_bips"],
                gates["annual_opportunity_hurdle_bips_per_capital_leg"],
                gates["gross_capital_legs"],
            ) != ("32", "1000", 2):
                raise ValueError("original cost contract differs")
            receipts = {row["name"]: row for row in prior["sources"]["responses"]}
            original_symbols = prior["symbol_results"]
            if (
                len(original_symbols) != 17
                or len({x["future_symbol"] for x in original_symbols}) != 17
            ):
                raise ValueError("original complete population differs")
            outcomes = []
            for symbol_result in original_symbols:
                symbol = symbol_result["future_symbol"]
                receipt = receipts["funding-" + symbol.lower()]
                raw_path = (ROOT / receipt["raw_path"]).resolve()
                raw_path.relative_to(RAW_BASE)
                history = normalize_history(
                    json.loads(read(raw_path, receipt["response_sha256"])), symbol
                )
                for role in ("training", "validation", "test"):
                    old = symbol_result["roles"][role]
                    first, last = (
                        old["first_funding_time_ms"],
                        old["last_funding_time_ms"],
                    )
                    selected = [row for row in history if first <= row["time"] <= last]
                    if (
                        len(selected) != old["observation_count"]
                        or selected[0]["time"] != first
                        or selected[-1]["time"] != last
                    ):
                        raise ValueError("original role endpoints/count differ")
                    earlier = [row for row in history if row["time"] < first]
                    if not earlier:
                        raise ValueError("pre-role reference mark absent")
                    metrics = cashflow_metrics(
                        selected,
                        earlier[-1]["mark"],
                        D(old["duration_days"]),
                        D("32"),
                        D(2),
                        D("1000"),
                    )
                    if D(metrics["original_unit_weight_funding_bps"]) != D(
                        old["gross_funding_bips"]
                    ):
                        raise ValueError(
                            "original gross funding reconstruction differs"
                        )
                    if abs(
                        D(metrics["original_capital_stress_bps"])
                        - D(old["capital_opportunity_hurdle_bips"])
                    ) > D("1e-20"):
                        raise ValueError(
                            "original capital stress reconstruction differs"
                        )
                    outcomes.append(
                        {
                            "symbol": symbol,
                            "role": role,
                            "row_count": len(selected),
                            "first_time_ms": first,
                            "last_time_ms": last,
                            "reference_time_ms": earlier[-1]["time"],
                            "duration_days": old["duration_days"],
                            "metrics": metrics,
                        }
                    )
            read(Path(__file__))
            result = {
                "schema_version": 1,
                "classification": plan["classification"],
                "accepted_edge": False,
                "profitability_claim": False,
                "new_requests": 0,
                "old_acceptance_rerun": False,
                "reference_is_executable_entry": False,
                "basis_execution_liquidation_and_stability_revalidated": False,
                "bindings": bindings,
                "symbol_count": 17,
                "role_count": len(outcomes),
                "rows": outcomes,
            }
            result["result_sha256"] = canonical(result, "result_sha256")
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            record("completed_offline_accounting_audit_no_promotion")
            print(
                json.dumps(
                    {
                        "symbol_count": 17,
                        "role_count": len(outcomes),
                        "result_sha256": result["result_sha256"],
                    }
                )
            )
        except Exception:
            record("terminal_offline_audit_failure_no_retry")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
