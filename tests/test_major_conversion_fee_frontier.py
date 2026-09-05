from decimal import Decimal as D
import json

import pytest

from tools.capture_binance_triangle_window import PAIRS
from tools.review_major_conversion_fee_frontier import compare_rates, routes


def test_uniform_extra_fee_break_even_and_direct_comparator():
    x = compare_rates(D(1), D(1), D("1.01"), D(0), D(0), D(0))
    assert x.gross_incremental_bips == 100
    assert abs(x.uniform_fee_break_even_bips - D("99.00990099009900990099009901")) < D(
        "1e-20"
    )
    assert (
        compare_rates(
            D(1), D(1), D("1.01"), D(".01"), D(".01"), D(".01")
        ).net_incremental_bips
        == -1
    )


def test_asymmetric_fees_can_reverse_a_gross_ranking():
    x = compare_rates(D(1), D(1), D("1.01"), D(0), D(".01"), D(".01"))
    assert x.net_incremental_bips == D("-100.99")
    y = compare_rates(D(1), D(1), D(".999"), D(".01"), D(0), D(0))
    assert y.gross_incremental_bips < 0 < y.net_incremental_bips


@pytest.mark.parametrize("bad", [D(0), D(-1), D("NaN"), D("Infinity"), 1.0])
def test_invalid_rates_reject(bad):
    with pytest.raises(ValueError):
        compare_rates(bad, D(1), D(1), D(0), D(0), D(0))


@pytest.mark.parametrize("bad", [D(-1), D(1), D("NaN"), D("Infinity"), 0.0])
def test_invalid_fees_reject(bad):
    with pytest.raises(ValueError):
        compare_rates(D(1), D(1), D(1), bad, D(0), D(0))


def test_exhaustive_twenty_four_routes_and_uniform_fee_penalty():
    unit = {"BTC": D(100), "ETH": D(10), "SOL": D(1), "USDT": D(1)}
    rows = [
        {
            "symbol": s,
            "bidPrice": str(unit[b] / unit[q]),
            "askPrice": str(unit[b] / unit[q]),
            "bidQty": "1000",
            "askQty": "1000",
        }
        for s, (b, q) in PAIRS.items()
    ]
    values = routes(json.dumps(rows).encode(), ("0", "1.2", "2.4", "10"))
    assert len(values) == len({x["route"] for x in values}) == 24
    assert len({x["direct"] for x in values}) == 12
    assert all(D(x["gross_incremental_bips"]) == 0 for x in values)
    for value in values:
        for fee, profit in value["net_incremental_bips_by_uniform_fee"].items():
            assert D(profit) == -D(fee)
