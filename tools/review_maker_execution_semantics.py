"""Offline counterexamples, not a fill estimator or an economic backtest.

Preserve the frozen full-fill support implementation. Demonstrate why its
Boolean labels cannot establish zero inventory or paired net cash profit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path

from simple_ai_trading.queue_censored_actions import build_passive_fill_result


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/simple_ai_trading/queue_censored_actions.py",
    "src/simple_ai_trading/paper_execution.py",
    "src/simple_ai_trading/polymarket_liquidity_rewards.py",
    "src/simple_ai_trading/polymarket_maker_rebates.py",
    "tools/probe_round58_two_sided_maker_support.py",
    "tools/publish_round58_two_sided_maker_feasibility.py",
    "docs/model-research/polymarket/crypto-maker-rebate-economics-v1.json",
    "tools/review_maker_execution_semantics.py",
    "docs/review/2026-09-04/maker-execution-review-plan.json",
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def matched_cash_ledger(
    *,
    buy_price: Decimal,
    sell_price: Decimal,
    bought_base: Decimal,
    sold_base: Decimal,
    fees_quote: Decimal,
) -> dict[str, str]:
    """Spot cash/base identity; residual inventory is deliberately unvalued.

    Inputs must be actual net delivered quantities for an execution ledger, or
    explicitly illustrative quantities for a counterexample. A zero cash flow
    with positive residual base is not a completed round-trip profit. For a
    linear derivative this is only the constant term in marked PnL, not the
    venue's margin cash-transfer ledger; residual mark exposure remains.
    """
    values = (buy_price, sell_price, bought_base, sold_base, fees_quote)
    if any(not isinstance(x, Decimal) or not x.is_finite() for x in values):
        raise ValueError("ledger inputs must be finite Decimals")
    if min(buy_price, sell_price) <= 0 or min(bought_base, sold_base, fees_quote) < 0:
        raise ValueError("prices must be positive and quantities/costs nonnegative")
    with localcontext() as context:
        context.prec = 50
        cash = sold_base * sell_price - bought_base * buy_price - fees_quote
        residual = bought_base - sold_base
    return {"net_cash_quote": str(cash), "residual_base": str(residual)}


def _support(*, price: float, print_quantity: float):
    return build_passive_fill_result(
        arrival_time_ms=[1000],
        placement_price=[price],
        queue_ahead_quantity=[5.0],
        buyer_is_maker=True,
        order_notional_quote=1000.0,
        trade_id=[1],
        trade_time_ms=[1100],
        trade_price=[price],
        trade_quantity=[print_quantity],
        trade_buyer_is_maker=[True],
    )


def build_report() -> dict[str, object]:
    # All numbers below are synthetic, selected to expose semantics, not alpha.
    partial = _support(price=100.0, print_quantity=14.0)
    none = _support(price=100.0, print_quantity=4.0)
    full = _support(price=100.0, print_quantity=15.0)
    projection = lambda r: {  # noqa: E731
        "full_fill": bool(r.filled[0]),
        "reported_matching_trade_count": int(r.matching_trade_count[0]),
        "reported_printed_quantity_through_fill": float(
            r.printed_quantity_through_fill[0]
        ),
    }
    d = Decimal
    source = json.loads((ROOT / SOURCES[6]).read_bytes())
    expected = source.pop("result_sha256")
    if hashlib.sha256(canonical(source)).hexdigest() != expected:
        raise ValueError("retained Polymarket example self-hash differs")
    example = source["example"]
    gain = d(example["settlement_both_fill_gross_profit"])
    worst_loss = d(example["maximum_orphan_settlement_loss_without_rebate_credit"])
    with localcontext() as context:
        context.prec = 50
        threshold = worst_loss / (gain + worst_loss)
    report = {
        "schema_version": "maker-execution-semantics-review-v1",
        "status": "synthetic_counterexamples_and_retained_example_only",
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in SOURCES
        },
        "equal_quote_notional_example": {
            "buy_price": "100",
            "sell_price": "125",
            "quote_notional_per_side": "1000",
            "bought_base": "10",
            "sold_base": "8",
            **matched_cash_ledger(
                buy_price=d(100),
                sell_price=d(125),
                bought_base=d(10),
                sold_base=d(8),
                fees_quote=d(0),
            ),
            "residual_liquidation_value": None,
        },
        "equal_base_example": matched_cash_ledger(
            buy_price=d(100),
            sell_price=d(125),
            bought_base=d(8),
            sold_base=d(8),
            fees_quote=d(0),
        ),
        "full_fill_censoring": {
            "queue_ahead": 5,
            "own_quantity": 10,
            "four_matching_printed_units": projection(none),
            "fourteen_matching_printed_units": projection(partial),
            "fifteen_matching_printed_units": projection(full),
            "fourteen_units_implied_partial_under_same_fifo_assumption": 9,
            "owned_fill_claim": False,
        },
        "polymarket_conditional_stress_break_even": {
            "source_example_not_market_observation": True,
            "both_fill_gross_gain": str(gain),
            "assumed_orphan_loss_equal_to_worst_settlement_loss": str(worst_loss),
            "completion_probability_must_exceed": str(threshold),
            "conditioning": "at_least_one_full_fill_binary_abstraction",
            "actual_completion_probability": None,
            "actual_conditional_orphan_mean_loss": None,
            "incremental_costs_and_rewards": "excluded_not_assumed_zero_in_reality",
        },
        "venue_requests": 0,
        "historical_outcomes_opened": False,
        "model_trained": False,
        "profitability_claim": False,
        "accepted_edge": False,
        "trading_authority": False,
    }
    report["result_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.journal.exists():
        raise FileExistsError("one-use review output or journal already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with args.journal.open("x", encoding="utf-8", newline="\n") as journal:

        def record(payload: dict[str, object]) -> None:
            journal.write(
                json.dumps(
                    {
                        "at_utc": datetime.now(timezone.utc).isoformat(),
                        **payload,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            journal.flush()
            os.fsync(journal.fileno())

        record({"phase": "started", "venue_requests": 0})
        try:
            result = build_report()
            with args.output.open("xb") as output:
                output.write(
                    json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"
                )
                output.flush()
                os.fsync(output.fileno())
            record({"phase": "complete", "result_sha256": result["result_sha256"]})
        except Exception as error:
            record({"phase": "failed", "error_type": type(error).__name__})
            raise
    print(
        json.dumps(
            {"status": result["status"], "result_sha256": result["result_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
