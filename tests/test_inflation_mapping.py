"""Offline provenance and conditional algebra; not trading performance tests."""

from decimal import Decimal as D
import json

import pytest

from tools.review_inflation_mapping import BASE, aggregation, build_review, sha


def test_review_reconstructs_without_market_prices():
    value = json.loads((BASE / "review.json").read_bytes())
    expected = value.pop("result_sha256")
    assert (
        sha(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        == expected
    )
    value.pop("created_at_utc")
    assert value == build_review()
    assert value["market_counts"] == [12, 9]
    assert not value["prices_inspected"] and not value["accepted_edge"]


def test_same_unadjusted_headline_does_not_fix_adjusted_headline():
    a = aggregation((D(90), D(110)), (D("0.9"), D("1.1")))
    b = aggregation((D(110), D(90)), (D("0.9"), D("1.1")))
    assert a[0] == b[0] == 200
    assert a[1] == 200 and b[1] > 204


def test_equal_factors_are_the_special_single_factor_case():
    for values in ((D(90), D(110)), (D(110), D(90))):
        nsa, sa = aggregation(values, (D(2), D(2)))
        assert sa == nsa / 2


def test_positive_component_ratio_stays_in_reciprocal_factor_hull():
    factors = (D("0.9"), D("1.1"))
    for x in range(1, 200):
        nsa, sa = aggregation((D(x), D(200 - x)), factors)
        assert min(1 / f for f in factors) <= sa / nsa <= max(1 / f for f in factors)


@pytest.mark.parametrize(
    "values,factors",
    [
        ([], []),
        ([D(1)], []),
        ([D(1)], [D(0)]),
        ([D("NaN")], [D(1)]),
        ([D(1)], [D("Infinity")]),
    ],
)
def test_undefined_aggregation_is_rejected(values, factors):
    with pytest.raises(ValueError):
        aggregation(values, factors)
