from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

import pytest

from simple_ai_trading.settlement_tracking import (
    SettlementObservation,
    evaluate_linear_settlement_tracking,
)


def _inputs():
    return dict(
        quantity=D("1"),
        spot_entry_price=D("100"),
        futures_entry_price=D("101"),
        total_cost_quote=D("0.2"),
        declared_timestamps_ms=(1, 2),
        observations=(
            SettlementObservation(1, D("90"), D("90"), D("90"), D("1"), D("0.5")),
            SettlementObservation(2, D("110"), D("110"), D("110"), D("1"), D("0.5")),
        ),
        spot_price_lower_bound=D("90"),
        spot_price_upper_bound=D("110"),
    )


@pytest.mark.parametrize("prices", [("90", "110"), ("110", "90"), ("100", "100")])
def test_matched_weights_remove_common_price_timing_effect(prices):
    inputs = _inputs()
    inputs["observations"] = tuple(
        replace(
            row,
            index_price=D(price),
            spot_reference_price=D(price),
            spot_execution_price=D(price),
        )
        for row, price in zip(inputs["observations"], prices, strict=True)
    )
    result = evaluate_linear_settlement_tracking(**inputs)
    assert result.net_cashflow_quote == result.direct_cashflow_quote == D("0.8")
    assert result.timing_tracking_quote == result.index_tracking_quote == 0
    assert result.conditional_timing_loss_bound_quote == 0
    assert result.qualified_edge is False


@pytest.mark.parametrize(
    "weights,expected", [(("1", "0"), "-9.2"), (("0", "1"), "10.8")]
)
def test_point_exit_timing_can_dominate_entry_basis(weights, expected):
    inputs = _inputs()
    inputs["observations"] = tuple(
        replace(
            row,
            spot_exit_weight=D(weight),
            spot_execution_price=row.spot_reference_price if D(weight) else None,
        )
        for row, weight in zip(inputs["observations"], weights, strict=True)
    )
    result = evaluate_linear_settlement_tracking(**inputs)
    assert result.net_cashflow_quote == result.direct_cashflow_quote == D(expected)
    assert result.conditional_timing_loss_bound_quote == D("10")
    assert (
        abs(result.timing_tracking_quote) == result.conditional_timing_loss_bound_quote
    )


def test_matched_timing_does_not_remove_index_basis_or_costs():
    inputs = _inputs()
    inputs["observations"] = tuple(
        replace(row, index_price=row.index_price + 2) for row in inputs["observations"]
    )
    result = evaluate_linear_settlement_tracking(**inputs)
    assert result.index_tracking_quote == -2
    assert result.timing_tracking_quote == 0
    assert result.net_cashflow_quote == result.direct_cashflow_quote == D("-1.2")


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantity", D("NaN")),
        ("quantity", D("0")),
        ("quantity", 1.0),
        ("total_cost_quote", D("-1")),
        ("total_cost_quote", D("Infinity")),
        ("spot_price_lower_bound", D("111")),
        ("spot_price_upper_bound", D("100")),
        ("declared_timestamps_ms", (1,)),
        ("declared_timestamps_ms", (1, 1)),
        ("declared_timestamps_ms", (2, 1)),
        ("declared_timestamps_ms", (True, 2)),
        ("declared_timestamps_ms", ()),
        ("observations", ()),
    ],
)
def test_invalid_or_incomplete_inputs_reject(field, value):
    inputs = _inputs()
    inputs[field] = value
    with pytest.raises(ValueError):
        evaluate_linear_settlement_tracking(**inputs)


@pytest.mark.parametrize(
    "change",
    [
        {"settlement_weight_units": D("0")},
        {"spot_exit_weight": D("-0.1")},
        {"spot_exit_weight": D("0.4")},
        {"settlement_weight_units": D("-0.6")},
        {"index_price": D("NaN")},
        {"spot_execution_price": D("0")},
        {"timestamp_ms": True},
    ],
)
def test_invalid_observation_and_quantity_conservation_reject(change):
    inputs = _inputs()
    inputs["observations"] = (
        replace(inputs["observations"][0], **change),
        inputs["observations"][1],
    )
    with pytest.raises(ValueError):
        evaluate_linear_settlement_tracking(**inputs)


def test_price_impact_is_separate_from_reference_timing_and_fees():
    inputs = _inputs()
    inputs["observations"] = tuple(
        replace(row, spot_execution_price=row.spot_reference_price - 3)
        for row in inputs["observations"]
    )
    result = evaluate_linear_settlement_tracking(**inputs)
    assert result.timing_tracking_quote == result.index_tracking_quote == 0
    assert result.execution_tracking_quote == -3
    assert result.net_cashflow_quote == result.direct_cashflow_quote == D("-2.2")


@pytest.mark.parametrize("exit_weight,execution", [("0.5", None), ("0", D("90"))])
def test_missing_or_fictitious_execution_is_rejected(exit_weight, execution):
    inputs = _inputs()
    inputs["observations"] = (
        replace(
            inputs["observations"][0],
            spot_exit_weight=D(exit_weight),
            spot_execution_price=execution,
        ),
        inputs["observations"][1],
    )
    with pytest.raises(ValueError):
        evaluate_linear_settlement_tracking(**inputs)


def test_equal_one_second_samples_do_not_need_rounded_decimal_weights():
    inputs = _inputs()
    inputs["declared_timestamps_ms"] = tuple(range(1000, 1_800_001, 1000))
    inputs["observations"] = tuple(
        SettlementObservation(
            timestamp,
            D("100"),
            D("100"),
            D("100") if i == 1799 else None,
            D("1"),
            D("1") if i == 1799 else D("0"),
        )
        for i, timestamp in enumerate(inputs["declared_timestamps_ms"])
    )
    result = evaluate_linear_settlement_tracking(**inputs)
    assert result.settlement_price == 100
    assert result.net_cashflow_quote == result.direct_cashflow_quote == D("0.8")


def test_common_scaling_of_settlement_units_does_not_change_results():
    inputs = _inputs()
    expected = evaluate_linear_settlement_tracking(**inputs)
    inputs["observations"] = tuple(
        replace(row, settlement_weight_units=D("1800"))
        for row in inputs["observations"]
    )
    assert evaluate_linear_settlement_tracking(**inputs) == expected


def test_exact_rational_reconciliation_handles_nonterminating_average():
    inputs = _inputs()
    inputs["declared_timestamps_ms"] = (1, 2, 3)
    inputs["observations"] = (
        SettlementObservation(1, D("90"), D("90"), None, D("1"), D("0")),
        SettlementObservation(2, D("100"), D("100"), None, D("1"), D("0")),
        SettlementObservation(3, D("102"), D("102"), D("102"), D("1"), D("1")),
    )
    result = evaluate_linear_settlement_tracking(**inputs)
    assert result.net_cashflow_quote == result.direct_cashflow_quote
    assert D("5.466666") < result.net_cashflow_quote < D("5.466667")


def test_numerical_resource_contract_rejects_extreme_exponents():
    inputs = _inputs()
    inputs["quantity"] = D("1e1000000")
    with pytest.raises(ValueError, match="bounded"):
        evaluate_linear_settlement_tracking(**inputs)
