"""Offline economic counterexamples; no venue access or performance timing."""

from decimal import Decimal
import hashlib
import json

import pytest

from tools.review_maker_execution_semantics import (
    ROOT,
    build_report,
    canonical,
    matched_cash_ledger,
)


def test_equal_quote_notional_is_not_equal_base_inventory() -> None:
    result = matched_cash_ledger(
        buy_price=Decimal(100),
        sell_price=Decimal(125),
        bought_base=Decimal(10),
        sold_base=Decimal(8),
        fees_quote=Decimal(0),
    )
    assert result == {"net_cash_quote": "0", "residual_base": "2"}


def test_equal_base_closes_inventory_and_subtracts_fees_once() -> None:
    result = matched_cash_ledger(
        buy_price=Decimal(100),
        sell_price=Decimal(125),
        bought_base=Decimal(8),
        sold_base=Decimal(8),
        fees_quote=Decimal(3),
    )
    assert result == {"net_cash_quote": "197", "residual_base": "0"}


def test_sell_excess_is_not_silently_clamped_flat() -> None:
    result = matched_cash_ledger(
        buy_price=Decimal(100),
        sell_price=Decimal(125),
        bought_base=Decimal(8),
        sold_base=Decimal(10),
        fees_quote=Decimal(0),
    )
    assert result["residual_base"] == "-2"


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), True, None, 1.0])
def test_unknown_and_nonfinite_inputs_fail_closed(bad) -> None:
    with pytest.raises(ValueError, match="finite Decimals"):
        matched_cash_ledger(
            buy_price=bad,
            sell_price=Decimal(125),
            bought_base=Decimal(8),
            sold_base=Decimal(8),
            fees_quote=Decimal(0),
        )


def test_negative_cost_is_not_implicitly_credited_as_a_rebate() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        matched_cash_ledger(
            buy_price=Decimal(100),
            sell_price=Decimal(125),
            bought_base=Decimal(8),
            sold_base=Decimal(8),
            fees_quote=Decimal(-1),
        )


def test_retained_full_fill_labels_cannot_identify_partial_inventory() -> None:
    report = build_report()
    censoring = report["full_fill_censoring"]
    assert (
        censoring["four_matching_printed_units"]
        == censoring["fourteen_matching_printed_units"]
    )
    assert not censoring["fourteen_matching_printed_units"]["full_fill"]
    assert censoring["fifteen_matching_printed_units"]["full_fill"]
    assert censoring["fourteen_units_implied_partial_under_same_fifo_assumption"] == 9
    threshold = report["polymarket_conditional_stress_break_even"]
    assert (
        Decimal("0.96078")
        < Decimal(threshold["completion_probability_must_exceed"])
        < Decimal("0.96079")
    )
    assert threshold["actual_completion_probability"] is None
    assert not report["profitability_claim"]


def test_committed_report_reconstructs_and_source_bytes_match() -> None:
    report = json.loads(
        (ROOT / "docs/review/2026-09-04/maker-execution-review.json").read_bytes()
    )
    assert report == build_report()
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    expected = report.pop("result_sha256")
    assert hashlib.sha256(canonical(report)).hexdigest() == expected
