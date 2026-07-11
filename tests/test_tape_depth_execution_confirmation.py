from __future__ import annotations

from datetime import UTC, datetime

import pytest

from simple_ai_trading.cli import _build_parser
from simple_ai_trading.tape_depth_execution_confirmation import (
    TAPE_DEPTH_EXECUTION_PERIOD_SCHEMA_VERSION,
    _dataset_bounds_for_period,
    _with_fingerprint,
    aggregate_tape_depth_execution_confirmation,
)


_PERIODS = ("2023-07-08", "2023-10-21", "2024-02-05")
_DESIGN_SHA256 = "d" * 64


def _design() -> dict[str, object]:
    return {
        "confirmation_periods": list(_PERIODS),
        "availability_sha256": "a" * 64,
        "acceptance": {
            "maximum_liquidations": 0,
            "maximum_quote_rejection_rows": 0,
            "minimum_combined_executable_rows": 15,
            "minimum_combined_mean_net_return_bps_exclusive": 0.0,
            "minimum_combined_positive_net_rate_exclusive": 0.5,
            "minimum_forecast_candidate_periods": 2,
            "minimum_positive_mean_net_periods": 2,
            "required_completed_periods": 3,
        },
    }


def _period_report(
    period: str,
    net_returns: list[float],
    *,
    forecast_status: str = "research_candidate",
    quote_rejection: bool = False,
) -> dict[str, object]:
    rows: list[dict[str, object]] = [
        {
            "status": "executable",
            "rejection_reason": "",
            "net_return_bps": value,
        }
        for value in net_returns
    ]
    if quote_rejection:
        rows.append(
            {
                "status": "rejected",
                "rejection_reason": "stale_entry_quote",
                "net_return_bps": None,
            }
        )
    executable = len(net_returns)
    positive = sum(value > 0.0 for value in net_returns)
    mean_net = sum(net_returns) / executable if executable else 0.0
    payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_EXECUTION_PERIOD_SCHEMA_VERSION,
        "status": "complete",
        "period": period,
        "design_sha256": _DESIGN_SHA256,
        "selection_contaminated": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "liquidation_events": 0,
        "forecast": {"status": forecast_status},
        "execution": {
            "status": "after_cost_diagnostic_candidate",
            "metrics": {
                "selected_signal_rows": len(rows),
                "overlap_suppressed_rows": 0,
                "scheduled_signal_rows": len(rows),
                "executable_rows": executable,
                "rejected_quote_rows": int(quote_rejection),
                "rejected_participation_rows": 0,
                "mean_net_return_bps": mean_net,
                "positive_net_rate": positive / executable if executable else 0.0,
            },
        },
        "execution_rows": rows,
    }
    return _with_fingerprint(payload, "period_fingerprint")


def test_execution_confirmation_aggregates_exact_row_evidence() -> None:
    reports = [_period_report(period, [1.0, 2.0, 3.0, 4.0, 5.0]) for period in _PERIODS]

    result = aggregate_tape_depth_execution_confirmation(
        _design(),
        design_sha256=_DESIGN_SHA256,
        period_reports=reports,
    )

    assert result["status"] == "confirmed_after_cost_candidate"
    assert result["rejection_reasons"] == []
    assert result["actual"] == {
        "completed_periods": 3,
        "forecast_candidate_periods": 3,
        "positive_mean_net_periods": 3,
        "combined_executable_rows": 15,
        "combined_mean_net_return_bps": 3.0,
        "combined_positive_net_rate": 1.0,
        "quote_rejection_rows": 0,
        "liquidation_events": 0,
    }
    assert result["trading_authority"] is False
    assert result["profitability_claim"] is False


def test_execution_confirmation_rejects_cost_failure_and_quote_gap() -> None:
    reports = [
        _period_report(_PERIODS[0], [-1.0] * 5, quote_rejection=True),
        _period_report(_PERIODS[1], [-2.0] * 5),
        _period_report(_PERIODS[2], [1.0] * 5, forecast_status="rejected"),
    ]

    result = aggregate_tape_depth_execution_confirmation(
        _design(),
        design_sha256=_DESIGN_SHA256,
        period_reports=reports,
    )

    assert result["status"] == "rejected"
    assert set(result["rejection_reasons"]) == {
        "positive_mean_net_periods",
        "combined_mean_net_return_bps",
        "combined_positive_net_rate",
        "quote_rejection_rows",
    }


def test_execution_confirmation_recomputes_metrics_from_rows() -> None:
    reports = [_period_report(period, [1.0] * 5) for period in _PERIODS]
    reports[0]["execution_rows"][0]["net_return_bps"] = -20.0  # type: ignore[index]
    reports[0] = _with_fingerprint(reports[0], "period_fingerprint")

    with pytest.raises(ValueError, match="row evidence differs"):
        aggregate_tape_depth_execution_confirmation(
            _design(),
            design_sha256=_DESIGN_SHA256,
            period_reports=reports,
        )


class _Connection:
    def __init__(self, row: tuple[int, int, int]) -> None:
        self.row = row

    def execute(self, _query: str, _parameters: list[object]):
        return self

    def fetchone(self) -> tuple[int, int, int]:
        return self.row


class _Warehouse:
    def __init__(self, row: tuple[int, int, int]) -> None:
        self.connection = _Connection(row)

    def connect(self) -> _Connection:
        return self.connection


def test_execution_confirmation_period_bounds_are_causal_and_date_local() -> None:
    day_start = int(
        datetime(2023, 7, 8, tzinfo=UTC).timestamp() * 1_000
    )
    warehouse = _Warehouse((day_start, day_start + 86_399_000, 86_400))

    start_ms, end_ms, evidence = _dataset_bounds_for_period(
        warehouse,  # type: ignore[arg-type]
        symbol="BTCUSDT",
        period="2023-07-08",
        horizon_seconds=20,
        total_latency_ms=750,
    )

    assert start_ms == day_start + 901_000
    assert end_ms == day_start + 86_379_000
    assert evidence["trade_second_rows"] == 86_400


def test_cli_exposes_exact_bbo_confirmation_as_a_distinct_workflow() -> None:
    args = _build_parser().parse_args(["tape-depth-execution-confirm"])

    assert args.func.__name__ == "command_tape_depth_execution_confirm"
    assert args.design.endswith("confirmation-design.json")
    assert args.resume is False
