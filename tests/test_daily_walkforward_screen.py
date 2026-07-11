from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from simple_ai_trading.microstructure_walkforward import ActionTrace
from simple_ai_trading.microstructure_model import TradingMetrics
from tools.run_daily_walkforward_screen import (
    _aggregate_gate_reasons,
    _aggregate_traces,
    _selection_rank,
    load_daily_walkforward_design,
)


DAY_MS = 86_400_000
ROOT = Path(__file__).resolve().parents[1]


def _trace(day: int, pnls: tuple[float, ...], gross: tuple[float, ...]) -> ActionTrace:
    metrics = TradingMetrics(
        trades=len(pnls),
        total_net_bps=sum(pnls),
        mean_net_bps=sum(pnls) / max(1, len(pnls)),
        median_net_bps=0.0,
        win_rate=0.0,
        profit_factor=None,
        max_drawdown_bps=0.0,
        worst_trade_bps=min(pnls, default=0.0),
        best_trade_bps=max(pnls, default=0.0),
        long_trades=len(pnls),
        short_trades=0,
        active_days=1 if pnls else 0,
        trades_per_active_day=float(len(pnls)),
    )
    return ActionTrace(
        metrics=metrics,
        gross_bps=gross,
        net_bps=pnls,
        sides=tuple(1 for _value in pnls),
        timestamps_ms=tuple(day * DAY_MS + index * 1_000 for index in range(len(pnls))),
        source_endpoint_indexes=tuple(range(len(pnls))),
    )


def test_aggregate_trace_counts_abstention_days_and_real_drawdown() -> None:
    aggregate = _aggregate_traces(
        [_trace(1, (10.0, -5.0), (22.0, 7.0)), _trace(3, (4.0,), (16.0,))],
        expected_days=(1, 2, 3),
    )

    assert aggregate["metrics"]["trades"] == 3
    assert aggregate["metrics"]["total_net_bps"] == 9.0
    assert aggregate["total_gross_bps"] == 45.0
    assert aggregate["abstention_days"] == 1
    assert aggregate["positive_day_ratio"] == 2 / 3
    assert aggregate["portfolio_claim"] is False


def test_aggregate_gates_and_selection_rank_are_risk_aware() -> None:
    aggregate = {
        "metrics": {
            "trades": 12,
            "total_net_bps": 20.0,
            "mean_net_bps": 2.0,
            "max_drawdown_bps": 8.0,
            "worst_trade_bps": -12.0,
        },
        "positive_day_ratio": 0.6,
    }
    gates = {
        "minimum_trades": 10,
        "minimum_total_net_bps": 0.0,
        "maximum_drawdown_bps": 50.0,
        "minimum_positive_day_ratio": 0.4,
        "minimum_worst_trade_bps": -30.0,
    }

    assert _aggregate_gate_reasons(aggregate, gates) == []
    assert _selection_rank(aggregate, 0.25) == (18.0, 2.0, 0.6, 12)

    failed = dict(aggregate)
    failed["metrics"] = dict(
        aggregate["metrics"], total_net_bps=-1.0, worst_trade_bps=-40.0
    )
    reasons = _aggregate_gate_reasons(failed, gates)
    assert "total_net_gate_failed" in reasons
    assert "worst_trade_gate_failed" in reasons


def test_aggregate_rejects_gross_net_count_drift() -> None:
    broken = _trace(1, (1.0,), (13.0,))
    broken = SimpleNamespace(
        net_bps=broken.net_bps,
        gross_bps=(),
        sides=broken.sides,
        timestamps_ms=broken.timestamps_ms,
    )
    try:
        _aggregate_traces([broken], expected_days=(1,))
    except ValueError as exc:
        assert "count drifted" in str(exc)
    else:
        raise AssertionError("gross/net count drift must be rejected")


def test_tracked_round15_design_is_hash_bound_to_historical_implementation() -> None:
    design, design_sha256 = load_daily_walkforward_design(
        ROOT
        / "docs/model-research/action-value/round-015-daily-walk-forward-design.json",
        require_current=False,
    )

    assert design_sha256 == (
        "4ff50e579bb036d3146a8ea01e8f502efea0b2a0445df8f187f612199ebbe43c"
    )
    assert design["implementation"]["commit"] == (  # type: ignore[index]
        "df8a1591fcd9a7fd1110e5a52679245da6cf3b61"
    )
    assert (
        design["evaluation"][  # type: ignore[index]
            "development_used_for_candidate_selection"
        ]
        is False
    )
    assert design["reserved_terminal"] == {
        "date": "2023-07-07",
        "included_in_dataset": False,
        "access_permitted": False,
    }
