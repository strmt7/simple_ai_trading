"""Generate synthetic decision examples; zero market or account access."""

from dataclasses import asdict, replace
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

from simple_ai_trading.completion_economics import CompletionState, compare_completion


def build_review() -> dict:
    """Bind explicit joint scenarios and implementation, not measured probabilities."""
    root = Path(__file__).resolve().parents[1]
    full = CompletionState("full", D(1), D(100), D(40), D(0), D(1), None, None)
    partial = replace(
        full,
        label="partial",
        opposite_net_shares=D(40),
        acquisition_cash=D(16),
        original_residual_net_bid=D("0.5"),
    )
    cases = {
        "positive_historical_profit_but_sale_better": (full,),
        "partial_completion_retains_original_exposure": (partial,),
        "positive_mean_with_downside": (
            replace(full, probability=D("0.9"), acquisition_cash=D(30)),
            replace(partial, probability=D("0.1")),
        ),
        "opposite_overfill_is_not_free": (
            replace(
                full,
                opposite_net_shares=D(110),
                acquisition_cash=D(44),
                opposite_residual_net_bid=D("0.3"),
            ),
        ),
    }
    rows = []
    for name, states in cases.items():
        inputs = dict(
            original_net_shares=D(100),
            liquidation_net_proceeds=D(65),
            historical_acquisition_cost=D(35),
            states=states,
        )
        result = compare_completion(**inputs)
        rows.append(
            {
                "case": name,
                "inputs": {**inputs, "states": [asdict(s) for s in states]},
                "result": asdict(result),
            }
        )
    sources = [
        "src/simple_ai_trading/completion_economics.py",
        "tools/review_completion_economics.py",
        "tests/test_completion_economics.py",
        "docs/review/2026-09-05/holding-yield-change-discovery-plan.json",
        "docs/review/2026-09-05/holding-yield-change-discovery-extraction.json",
    ]
    return {
        "schema_version": "completion-decision-synthetic-review-v1",
        "classification": "conditional_synthetic_arithmetic_not_edge_or_model_validation",
        "quote_unit": "synthetic_common_quote_unit",
        "valuation_horizon": "same hypothetical horizon for every cash flow and comparator; not an observed timestamp",
        "probabilities": "Illustrative supplied joint weights; not fitted, calibrated or empirically estimated",
        "rows": rows,
        "holding_yield_discovery": {
            "searches": 1,
            "program_change_proved": False,
            "retrieval": "Unrelated event pages; no holding program evidence. Does not prove absence of a change.",
            "refresh_gate_satisfied": False,
            "followup_market_or_source_requests": 0,
            "accidental_price_exposure": "All returned event pages excluded from prospective promotion based on this discovery. No prices used, no event follow-up, no economic screening claim.",
            "lesson": "This broad-domain dated query was ineffective; do not repeat it or count event metadata as program evidence. Prefer an independently identified official program announcement for a later material-change trigger.",
        },
        "historical_results_changed": False,
        "qualified_edge": False,
        "source_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
        },
    }


if __name__ == "__main__":
    print(
        json.dumps(
            build_review(), default=str, sort_keys=True, indent=2, allow_nan=False
        )
    )
