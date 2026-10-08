from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from simple_ai_trading.cross_asset_cost_data import MINUTE_MS, SYMBOLS, MinuteSeries
from simple_ai_trading.derivatives_hurdle_data import FundingState
from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_inventory import (
    InventoryCashPath,
    replay_fixed_base_inventory,
)
from simple_ai_trading.funding_cash_labels import FundingCashLabelSeries
from simple_ai_trading.stateful_cash_replay import replay_stateful_fixed_base_cash
from simple_ai_trading.stateful_position_policy import stateful_position_schedule
from simple_ai_trading.stateful_turnover_model import (
    StatefulHourlyDataset,
    replay_always_long,
)


def path(*, positions=(1, 1), prices=(100, 200, 100), events=()) -> InventoryCashPath:
    return InventoryCashPath(
        "BTCUSDT",
        (10, 20, 30),
        tuple(Fraction(value) for value in prices),
        positions,
        Fraction(100),
        Fraction(6, 10_000),
        "a" * 64,
        events,
        tuple((event.funding_time_ms, event.rate) for event in events),
        0,
        40,
        "b" * 64,
    )


def event(time, *, rate="0.01", mark=150):
    return LinearFundingSettlement(
        "BTCUSDT", "USDT", time, Fraction(rate), Fraction(mark), "c" * 64
    )


def test_fixed_base_round_trip_retains_units_and_charges_actual_quote() -> None:
    replay = replay_fixed_base_inventory(path())
    assert [row.signed_base_quantity for row in replay.intervals] == [1, 1]
    assert [row.entry_quantity_change for row in replay.intervals] == [1, 0]
    assert replay.total_price_cash == 0
    assert replay.total_traded_quote == 200
    assert replay.total_execution_cost == Fraction(3, 25)
    assert (
        replay.total_net_cash_lower == replay.total_net_cash_upper == Fraction(-3, 25)
    )


def test_reversal_cost_uses_actual_quantity_not_sign_units() -> None:
    replay = replay_fixed_base_inventory(path(positions=(1, -1)))
    assert [row.signed_base_quantity for row in replay.intervals] == [
        1,
        Fraction(-1, 2),
    ]
    assert replay.intervals[1].entry_quantity_change == Fraction(-3, 2)
    assert replay.intervals[1].entry_traded_quote == 300
    assert replay.total_traded_quote == 450
    assert replay.total_price_cash == 150
    assert replay.total_execution_cost == Fraction(27, 100)


@pytest.mark.parametrize(
    "position,expected", [(1, Fraction(-3, 2)), (-1, Fraction(3, 2)), (0, Fraction(0))]
)
def test_continuous_shared_boundary_event_is_paid_once(position, expected) -> None:
    replay = replay_fixed_base_inventory(
        path(positions=(position, position), events=(event(20),))
    )
    assert len(replay.funding_events) == 1
    payment = replay.funding_events[0]
    assert not payment.boundary_quantity_uncertain
    assert payment.cash_lower == payment.cash_upper == expected


@pytest.mark.parametrize("time", [10, 30])
@pytest.mark.parametrize(
    "rate,low,high", [("0.01", Fraction(-3, 2), 0), ("-0.01", 0, Fraction(3, 2))]
)
def test_entry_and_terminal_boundary_enclose_uncertain_entitlement(
    time, rate, low, high
) -> None:
    replay = replay_fixed_base_inventory(path(events=(event(time, rate=rate),)))
    payment = replay.funding_events[0]
    assert payment.boundary_quantity_uncertain
    assert (payment.cash_lower, payment.cash_upper) == (low, high)


def test_reversal_boundary_encloses_old_partial_flat_and_new_once() -> None:
    replay = replay_fixed_base_inventory(path(positions=(1, -1), events=(event(20),)))
    payment = replay.funding_events[0]
    assert (payment.quantity_before, payment.quantity_after) == (1, Fraction(-1, 2))
    assert (payment.cash_lower, payment.cash_upper) == (Fraction(-3, 2), Fraction(3, 4))
    assert len(replay.funding_events) == 1


def test_only_path_events_paid_and_flat_close_has_no_terminal_cost() -> None:
    replay = replay_fixed_base_inventory(
        path(positions=(1, 0), events=(event(5), event(15), event(35)))
    )
    assert [payment.funding_time_ms for payment in replay.funding_events] == [15]
    assert replay.intervals[-1].terminal_traded_quote == 0
    assert replay.total_traded_quote == 300


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "BNBUSDT"},
        {"boundary_time_ms": (10, 10, 30)},
        {"boundary_time_ms": (True, 20, 30)},
        {"boundary_price": (Fraction(100), Fraction(0), Fraction(100))},
        {"boundary_price": (100.0, 200.0, 100.0)},
        {"desired_position": (1, True)},
        {"desired_position": ()},
        {"desired_position": [1, 1]},
        {"opening_quote_notional": Fraction(0)},
        {"one_way_cost_fraction": Fraction(-1)},
        {"coverage_start_ms": 11},
        {"coverage_end_exclusive_ms": 30},
        {"price_source_sha256": "not-evidence"},
        {"population_certificate_sha256": "x" * 64},
        {"settlements": (event(15),)},
        {"settlements": (event(40),), "expected_events": ((40, Fraction("0.01")),)},
    ],
)
def test_inventory_contract_rejects_invalid_or_incomplete_inputs(change) -> None:
    with pytest.raises(ValueError):
        replace(path(), **change)


def test_replay_requires_validated_path() -> None:
    with pytest.raises(ValueError, match="validated"):
        replay_fixed_base_inventory(None)


def inputs():
    start = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
    features = np.zeros((6, 1), dtype=np.float32)
    dataset = StatefulHourlyDataset(
        ("synthetic",),
        features,
        features,
        np.repeat(start + np.arange(2) * 60 * MINUTE_MS, 3),
        np.tile(np.arange(3), 2),
        np.repeat([10_000.0, -5_000.0], 3),
        np.zeros(6),
        None,
        "d" * 64,
    )
    boundaries = start + np.array([1, 61, 121]) * MINUTE_MS
    zeros = np.zeros(3)
    panel = {
        symbol: MinuteSeries(
            symbol,
            boundaries,
            np.array([100.0, 200.0, 100.0]),
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        )
        for symbol in SYMBOLS
    }
    cash = {
        symbol: FundingCashLabelSeries(
            symbol, (), (), start, start + 122 * MINUTE_MS, "b" * 64
        )
        for symbol in SYMBOLS
    }
    states = {
        symbol: FundingState(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        )
        for symbol in SYMBOLS
    }
    kwargs = dict(
        price_panel=panel,
        funding=states,
        funding_cash=cash,
        initial_quote_capital=Fraction(300),
        one_way_cost_fraction=Fraction(6, 10_000),
        mode="long_only",
        price_source_sha256="a" * 64,
    )
    return dataset, np.full(6, 13.0), kwargs


def test_reported_compounded_return_is_not_the_prior_array_sum() -> None:
    dataset, _, _ = inputs()
    legacy = replay_always_long(dataset, cost_scenario="synthetic", cost_bps=6, seed=1)
    assert np.sum(legacy.portfolio_return_bps) == pytest.approx(4_988)
    assert legacy.metrics["total_net_return_fraction"] == pytest.approx(-0.00149964)
    assert (1 + 0.9994) * (1 - 0.5006) - 1 == pytest.approx(-0.00149964)


def test_forward_replay_uses_cash_reference_equity_not_scalar_targets() -> None:
    dataset, predictions, kwargs = inputs()
    result = replay_stateful_fixed_base_cash(dataset, predictions, **kwargs)
    assert result.positions == ((1, 1, 1), (1, 1, 1))
    assert (
        result.total_return_lower == result.total_return_upper == Fraction(-12, 10_000)
    )
    assert result.quote_equity_lower == (
        Fraction(300),
        Fraction("599.82"),
        Fraction("299.64"),
    )
    assert sum(result.portfolio_interval_cash_lower) == Fraction("-0.36")
    assert not result.modeled_capital_exhausted and not result.financially_qualified
    tampered_targets = replace(
        dataset, signed_pre_transition_utility_bps=np.full(6, np.nan)
    )
    assert (
        replay_stateful_fixed_base_cash(tampered_targets, predictions, **kwargs)
        == result
    )
    changed = replay_stateful_fixed_base_cash(
        dataset, predictions, **{**kwargs, "price_source_sha256": "e" * 64}
    )
    assert changed.replay_input_sha256 != result.replay_input_sha256
    assert changed.total_return_lower == result.total_return_lower


def test_forward_replay_tracks_mark_cash_and_identity() -> None:
    dataset, predictions, kwargs = inputs()
    plain = replay_stateful_fixed_base_cash(dataset, predictions, **kwargs)
    for symbol in SYMBOLS:
        time = int(dataset.decision_time_ms[0]) + 61 * MINUTE_MS
        payment = LinearFundingSettlement(
            symbol, "USDT", time, Fraction("0.01"), Fraction(150), "c" * 64
        )
        source = kwargs["funding_cash"][symbol]
        kwargs["funding_cash"][symbol] = replace(
            source, settlements=(payment,), expected_events=((time, payment.rate),)
        )
        kwargs["funding"][symbol] = replace(
            kwargs["funding"][symbol],
            event_time_ms=np.array([time]),
            event_rate=np.array([0.01]),
        )
    result = replay_stateful_fixed_base_cash(dataset, predictions, **kwargs)
    assert result.quote_equity_lower[-1] == Fraction("295.14")
    assert result.replay_input_sha256 != plain.replay_input_sha256
    assert all(len(ledger.funding_events) == 1 for ledger in result.symbol_ledgers)


@pytest.mark.parametrize(
    "failure",
    [
        "symbols",
        "clock",
        "missing_price",
        "price",
        "price_symbol",
        "cash_scope",
        "population",
        "missing_cash",
        "forecast",
        "cost",
        "capital",
        "digest",
    ],
)
def test_forward_replay_rejects_misaligned_or_invalid_inputs(failure) -> None:
    dataset, predictions, kwargs = inputs()
    first = SYMBOLS[0]
    if failure == "symbols":
        dataset = replace(dataset, symbol_index=np.tile([1, 0, 2], 2))
    elif failure == "clock":
        clocks = dataset.decision_time_ms.copy()
        clocks[-3:] += MINUTE_MS
        dataset = replace(dataset, decision_time_ms=clocks)
    elif failure == "missing_price":
        kwargs["price_panel"][first] = replace(
            kwargs["price_panel"][first],
            open_time_ms=kwargs["price_panel"][first].open_time_ms + 1,
        )
    elif failure == "price":
        kwargs["price_panel"][first] = replace(
            kwargs["price_panel"][first], open=np.array([100.0, np.nan, 100.0])
        )
    elif failure == "price_symbol":
        kwargs["price_panel"][first] = replace(
            kwargs["price_panel"][first], symbol=SYMBOLS[1]
        )
    elif failure == "cash_scope":
        kwargs["funding_cash"][first] = replace(
            kwargs["funding_cash"][first],
            coverage_end_exclusive_ms=int(dataset.decision_time_ms[-1]),
        )
    elif failure == "population":
        kwargs["funding"][first] = replace(
            kwargs["funding"][first],
            event_time_ms=np.array([int(dataset.decision_time_ms[0])]),
            event_rate=np.array([0.01]),
        )
    elif failure == "missing_cash":
        kwargs["funding_cash"].pop(first)
    elif failure == "forecast":
        predictions[0] = np.inf
    elif failure == "cost":
        kwargs["one_way_cost_fraction"] = Fraction(0)
    elif failure == "capital":
        kwargs["initial_quote_capital"] = Fraction(-1)
    else:
        kwargs["price_source_sha256"] = "fake"
    with pytest.raises(ValueError):
        replay_stateful_fixed_base_cash(dataset, predictions, **kwargs)


def test_shared_policy_strict_hurdles_reversal_and_forced_hold() -> None:
    result = stateful_position_schedule(
        np.array([[12.0], [13.0], [-24.0], [-25.0], [-30.0], [-30.0]]),
        mode="long_short",
        cost_bps=6,
        maximum_holding_hours=2,
        cost_filter_multiplier=2,
    )
    assert result.positions[:, 0].tolist() == [0, 1, 1, 0, -1, -1]
    assert result.reasons[:, 0].tolist() == [0, 1, 0, 4, 1, 0]
    assert result.holding_durations == (2, 2)
    reverse = stateful_position_schedule(
        np.array([[13.0], [-25.0]]),
        mode="long_short",
        cost_bps=6,
        maximum_holding_hours=24,
        cost_filter_multiplier=2,
    )
    assert reverse.reasons[:, 0].tolist() == [1, 3]
    assert reverse.transitions[:, 0].tolist() == [1, 2]
    close = stateful_position_schedule(
        np.array([[13.0], [-13.0]]),
        mode="long_only",
        cost_bps=6,
        maximum_holding_hours=24,
        cost_filter_multiplier=2,
    )
    assert close.reasons[:, 0].tolist() == [1, 2]


def test_exhausted_reference_equity_is_flagged_not_presented_as_protection() -> None:
    dataset, predictions, kwargs = inputs()
    kwargs["mode"] = "long_short"
    predictions[:] = -25
    first = SYMBOLS[0]
    for symbol in SYMBOLS:
        kwargs["price_panel"][symbol] = replace(
            kwargs["price_panel"][symbol], open=np.array([100.0, 300.0, 300.0])
        )
    result = replay_stateful_fixed_base_cash(dataset, predictions, **kwargs)
    assert result.modeled_capital_exhausted and not result.financially_qualified
    assert result.quote_equity_lower[-1] < 0
    altered = dict(kwargs)
    altered["price_panel"] = dict(kwargs["price_panel"])
    altered["price_panel"][first] = replace(
        kwargs["price_panel"][first], open=np.array([100.0, 300.0, 301.0])
    )
    assert (
        replay_stateful_fixed_base_cash(
            dataset, predictions, **altered
        ).replay_input_sha256
        != result.replay_input_sha256
    )


@pytest.mark.parametrize(
    "failure",
    [
        "empty_evaluation",
        "incomplete_groups",
        "noninteger_decisions",
        "misaligned_times",
        "wrong_prediction_shape",
        "duplicate_prices",
        "empty_prices",
        "outside_prices",
        "invalid_dataset_hash",
        "overflow_cost",
    ],
)
def test_additional_grid_and_numeric_contracts(failure) -> None:
    dataset, predictions, kwargs = inputs()
    first = SYMBOLS[0]
    if failure == "empty_evaluation":
        dataset = replace(dataset, decision_time_ms=dataset.decision_time_ms - 10**12)
    elif failure == "incomplete_groups":
        clocks = dataset.decision_time_ms.copy()
        clocks[0] -= 60 * MINUTE_MS
        dataset = replace(dataset, decision_time_ms=clocks)
    elif failure == "noninteger_decisions":
        dataset = replace(
            dataset, decision_time_ms=dataset.decision_time_ms.astype(float)
        )
    elif failure == "misaligned_times":
        clocks = dataset.decision_time_ms.copy()
        clocks[0] += 1
        dataset = replace(dataset, decision_time_ms=clocks)
    elif failure == "wrong_prediction_shape":
        predictions = predictions[:-1]
    elif failure == "duplicate_prices":
        clocks = kwargs["price_panel"][first].open_time_ms.copy()
        clocks[1] = clocks[0]
        kwargs["price_panel"][first] = replace(
            kwargs["price_panel"][first], open_time_ms=clocks
        )
    elif failure == "empty_prices":
        kwargs["price_panel"][first] = replace(
            kwargs["price_panel"][first],
            open_time_ms=np.array([], dtype=np.int64),
            open=np.array([]),
        )
    elif failure == "outside_prices":
        series = kwargs["price_panel"][first]
        kwargs["price_panel"][first] = replace(
            series, open_time_ms=series.open_time_ms[:-1], open=series.open[:-1]
        )
    elif failure == "invalid_dataset_hash":
        dataset = replace(dataset, dataset_sha256="fake")
    else:
        kwargs["one_way_cost_fraction"] = Fraction(10**400)
    with pytest.raises(ValueError):
        replay_stateful_fixed_base_cash(dataset, predictions, **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"cost_bps": np.nan},
        {"cost_bps": 0},
        {"cost_filter_multiplier": 0},
        {"maximum_holding_hours": True},
        {"mode": "invalid"},
        {"forecasts": np.empty((0, 3))},
        {"forecasts": np.array([[np.inf]])},
    ],
)
def test_shared_policy_contract_rejects_invalid_inputs(change) -> None:
    kwargs = dict(
        forecasts=np.zeros((2, 3)),
        mode="long_short",
        cost_bps=6,
        maximum_holding_hours=24,
        cost_filter_multiplier=2,
    )
    with pytest.raises(ValueError):
        stateful_position_schedule(**{**kwargs, **change})
