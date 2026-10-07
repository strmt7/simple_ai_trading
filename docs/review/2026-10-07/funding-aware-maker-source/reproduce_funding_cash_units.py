"""Reproduce the source-bound funding-unit mismatch; no market or account access."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from simple_ai_trading.barrier_payoff_data import (
    _funding_in_holding_window as barrier_funding,
)
from simple_ai_trading.derivatives_hurdle_data import (
    FundingState,
    _funding_in_holding_window as hurdle_funding,
)


@dataclass(frozen=True)
class CashUnitControl:
    entry_price: Fraction = Fraction(100)
    exit_price: Fraction = Fraction("100.505")
    settlement_mark: Fraction = Fraction(102)
    settlement_rate: Fraction = Fraction("0.005")


def reproduce(root: Path, control: CashUnitControl) -> dict[str, object]:
    """Compare the actual two helpers with independent fixed-base cash algebra."""
    expected_sources = {
        "src/simple_ai_trading/derivatives_hurdle_data.py": (
            "a5b68a74a1b5a8cd9ecb42ff1e68ead2fba51888111d09dba023b19d38880176"
        ),
        "src/simple_ai_trading/barrier_payoff_data.py": (
            "715aa811e635c3c836866383663563e37b6967a6486bc395a420be15fe34f62b"
        ),
        "src/simple_ai_trading/market_store.py": (
            "37862843b1f1a011c6b3ad0770b9582a0ffb70c4cd4485a0c34c3965724852c0"
        ),
    }
    for path, expected in expected_sources.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen reproduction source differs: {path}")
    zeros = np.zeros(1, dtype=np.float64)
    state = FundingState(
        event_time_ms=np.array([10], dtype=np.int64),
        event_rate=np.array([float(control.settlement_rate)]),
        event_interval_hours=np.array([8]),
        last_rate_bps=zeros,
        last_interval_hours=np.full(1, 8.0),
        age_minutes=zeros,
        settled_sum_24h_bps=zeros,
        settled_sum_72h_bps=zeros,
        settled_sum_168h_bps=zeros,
        event_mean_30_bps=zeros,
        event_zscore_30=zeros,
    )
    entry_time = np.array([0], dtype=np.int64)
    exit_time = np.array([20], dtype=np.int64)
    correct_debit = (
        control.settlement_rate * control.settlement_mark / control.entry_price * 10_000
    )
    gross_price = (control.exit_price / control.entry_price - 1) * 10_000
    rows = []
    for name, function in (
        ("derivatives_hurdle_data", hurdle_funding),
        ("barrier_payoff_data", barrier_funding),
    ):
        proxy = Fraction(str(float(function(state, entry_time, exit_time)[0])))
        assert proxy == 50 and correct_debit == 51
        assert gross_price - proxy > 0 > gross_price - correct_debit
        assert -gross_price + proxy < 0 < -gross_price + correct_debit
        rows.append(
            {
                "consumer": name,
                "existing_rate_sum_bps": str(proxy),
                "correct_fixed_base_cash_debit_bps": str(correct_debit),
                "gross_long_price_return_bps": str(gross_price),
                "long_existing_bps": str(gross_price - proxy),
                "long_correct_bps": str(gross_price - correct_debit),
                "short_existing_bps": str(-gross_price + proxy),
                "short_correct_bps": str(-gross_price + correct_debit),
            }
        )
    return {
        "status": "confirmed_unrepaired_source_bound_cash_unit_mismatch",
        "source_sha256": expected_sources,
        "control_kind": "synthetic algebraic counterexample, not a market observation",
        "execution_fees": "excluded from control; not an after-all-cost edge",
        "rows": rows,
        "network_requests": 0,
        "credentials_used": False,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            reproduce(Path(__file__).resolve().parents[4], CashUnitControl()),
            sort_keys=True,
            allow_nan=False,
        )
    )
