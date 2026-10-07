"""Retained-event hedge break-even frontiers, not executable strategy returns."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from simple_ai_trading.funding_cash import (
    FundingEntitlement,
    LinearFundingSettlement,
    funding_cash_bounds_for_events,
    parse_binance_usdm_funding_history,
)
from tools.capture_public_source_contract import _canonical_hash, _root_path


DAY_MS = 86_400_000


def hedge_net_cash(
    *,
    base_quantity: Fraction,
    entry_spot: Fraction,
    exit_spot: Fraction,
    entry_future: Fraction,
    exit_future: Fraction,
    funding_cash: Fraction,
    total_cost: Fraction,
) -> Fraction:
    """Reconcile a supplied equal-base long-spot/short-linear hedge in USDT."""
    values = (base_quantity, entry_spot, exit_spot, entry_future, exit_future)
    if any(not isinstance(v, Fraction) or v <= 0 for v in values):
        raise ValueError("hedge quantities and prices must be positive Fractions")
    if (
        not isinstance(funding_cash, Fraction)
        or not isinstance(total_cost, Fraction)
        or total_cost < 0
    ):
        raise ValueError("hedge cash and nonnegative total costs must be Fractions")
    return (
        base_quantity * (exit_spot - entry_spot + entry_future - exit_future)
        + funding_cash
        - total_cost
    )


def frontier(
    events: Sequence[LinearFundingSettlement],
    *,
    end_ms: int,
    cost_reserves_bps: Sequence[Fraction],
    capital_aprs_bps: Sequence[Fraction],
    capital_multiple: Fraction,
) -> dict[str, object]:
    """Exclude the reference event; all remaining entitlement is hypothetical."""
    if (
        len(events) < 2
        or type(end_ms) is not int
        or not isinstance(capital_multiple, Fraction)
        or capital_multiple <= 0
        or not cost_reserves_bps
        or not capital_aprs_bps
        or any(
            not isinstance(v, Fraction) or v < 0
            for v in (*cost_reserves_bps, *capital_aprs_bps)
        )
    ):
        raise ValueError("frontier coverage or sensitivity contract is invalid")
    reference, *held_events = events
    if end_ms <= events[-1].funding_time_ms:
        raise ValueError("panel end must follow every supplied event")
    # Include the reference in alignment checks even though its payment is excluded.
    bounds = funding_cash_bounds_for_events(
        events,
        [FundingEntitlement.NOT_HELD] + [FundingEntitlement.HELD] * len(held_events),
        expected_symbol=reference.symbol,
        signed_base_quantity=Fraction(-1),
        entry_price=reference.settlement_mark,
    )
    uncertain = funding_cash_bounds_for_events(
        held_events,
        [FundingEntitlement.UNKNOWN] * len(held_events),
        expected_symbol=reference.symbol,
        signed_base_quantity=Fraction(-1),
        entry_price=reference.settlement_mark,
    )
    cash_path = [Fraction(0)]
    for event in held_events:
        cash_path.append(cash_path[-1] + event.settlement_mark * event.rate)
    peak = drawdown = Fraction(0)
    for cash in cash_path:
        peak = max(peak, cash)
        drawdown = max(drawdown, peak - cash)
    days = Fraction(end_ms - reference.funding_time_ms, DAY_MS)
    funding_bps = bounds.entry_relative_lower_bps
    sensitivities = []
    for reserve in cost_reserves_bps:
        for apr in capital_aprs_bps:
            capital_bps = apr * capital_multiple * days / 365
            sensitivities.append(
                {
                    "noncapital_cost_reserve_bps": str(reserve),
                    "annual_capital_cost_bps": str(apr),
                    "capital_cost_bps": str(capital_bps),
                    "maximum_basis_deterioration_bps_for_strict_positive_net": str(
                        funding_bps - reserve - capital_bps
                    ),
                }
            )
    return {
        "symbol": reference.symbol,
        "payment_asset": "USDT",
        "reference_event_ms": reference.funding_time_ms,
        "reference_mark_not_executable_entry": str(reference.settlement_mark),
        "reference_event_payment_excluded": True,
        "end_ms_exclusive": end_ms,
        "duration_days_at_reference": str(days),
        "supplied_events": len(events),
        "hypothetically_held_events": len(held_events),
        "funding_cash_usdt_per_base": str(bounds.cash_lower),
        "funding_bps_at_reference": str(funding_bps),
        "unknown_entitlement_cash_lower": str(uncertain.cash_lower),
        "unknown_entitlement_cash_upper": str(uncertain.cash_upper),
        "funding_only_prefund_usdt_per_base": str(-min(cash_path)),
        "funding_only_maximum_drawdown_usdt_per_base": str(drawdown),
        "capital_multiple_at_reference": str(capital_multiple),
        "sensitivities": sensitivities,
        "complete_hedge_return_observed": False,
        "margin_path_qualified": False,
        "profitability_claim": False,
    }


def _persist(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def review(contract_path: Path) -> dict[str, object]:
    """Consume a separately frozen offline study without another source request."""
    plan = json.loads(contract_path.read_bytes())
    if (
        _canonical_hash(plan, "contract_sha256") != plan["contract_sha256"]
        or contract_path.resolve() != _root_path(plan["contract_path"])
        or plan["schema_version"] != "native-funding-hedge-frontier-contract-v1"
        or plan["new_requests"] != 0
        or plan["accepted_edge"] is not False
        or plan["research_observation_count_effect"] != 0
    ):
        raise ValueError("offline frontier contract identity or authority differs")
    freeze = datetime.fromisoformat(plan["frozen_at_utc"].replace("Z", "+00:00"))
    if freeze.tzinfo is None or freeze > datetime.now(timezone.utc):
        raise ValueError("offline frontier freeze instant is invalid")
    for binding in plan["implementations"]:
        if (
            hashlib.sha256(_root_path(binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise ValueError("offline frontier implementation differs")
    result_path, journal_path = (_root_path(plan[k]) for k in ("output", "journal"))
    if result_path.exists() or journal_path.exists():
        raise FileExistsError("offline frontier study is already consumed")
    _persist(
        journal_path, {"phase": "intent", "contract_sha256": plan["contract_sha256"]}
    )
    try:
        by_symbol: dict[str, list[LinearFundingSettlement]] = {}
        for binding in plan["inputs"]:
            raw = _root_path(binding["path"]).read_bytes()
            parsed = parse_binance_usdm_funding_history(
                raw,
                expected_sha256=binding["sha256"],
                expected_symbol=binding["symbol"],
            )
            if len(parsed) != binding["rows"] or any(
                not plan["start_ms"] <= e.funding_time_ms < plan["end_ms"]
                for e in parsed
            ):
                raise ValueError("offline frontier source rows or window differ")
            by_symbol.setdefault(binding["symbol"], []).extend(parsed)
        if sorted(by_symbol) != plan["symbols"]:
            raise ValueError("offline frontier symbol population differs")
        rows = []
        for panel in plan["panels"]:
            for symbol in plan["symbols"]:
                events = [
                    e
                    for e in by_symbol[symbol]
                    if panel["start_ms"] <= e.funding_time_ms < panel["end_ms"]
                ]
                if len(events) != panel["expected_supplied_events_per_symbol"]:
                    raise ValueError(
                        "panel row count differs from frozen diagnostic population"
                    )
                row = frontier(
                    events,
                    end_ms=panel["end_ms"],
                    cost_reserves_bps=[Fraction(v) for v in plan["cost_reserves_bps"]],
                    capital_aprs_bps=[Fraction(v) for v in plan["capital_aprs_bps"]],
                    capital_multiple=Fraction(plan["capital_multiple"]),
                )
                row["panel"] = panel["name"]
                rows.append(row)
        result = {"passed": True, "panels": rows, "error": None}
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as failure:
        result = {"passed": False, "panels": [], "error": type(failure).__name__}
    result.update(
        {
            "schema_version": "native-funding-hedge-frontier-result-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": plan["contract_sha256"],
            "accepted_edge": False,
            "cash_labels_admitted": False,
            "profitability_claim": False,
            "independent_event_population_qualified": False,
            "owned_entitlement_qualified": False,
            "new_requests": 0,
            "research_observation_count_effect": 0,
        }
    )
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    _persist(result_path, result)
    with journal_path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(
            json.dumps(
                {
                    "phase": "completed",
                    "result_sha256": result["result_sha256"],
                    "passed": result["passed"],
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    result = review(parser.parse_args().contract)
    print(json.dumps({"passed": result["passed"], "panels": len(result["panels"])}))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
