"""Exact supplied-law controls; no venue data, fitting or order authority."""

from fractions import Fraction as F
from itertools import product

import pytest

from simple_ai_trading.partition_cover import (
    evaluate_partition_cover,
    separate_partition_cover,
    validate_partition_cover_certificate,
)


@pytest.mark.parametrize(
    "yes,no",
    [
        ([F(1, 5), F(1, 5), F(1, 5)], [None] * 3),
        ([F(1, 5), None, None], [F(1, 5), None, None]),
        ([None] * 3, [F(3, 5)] * 3),
        ([None, None, None, None], [F(1, 5), F(1, 5), None, F(2)]),
    ],
)
def test_each_infeasibility_case_returns_an_available_strict_cover(yes, no):
    result = separate_partition_cover(yes, no)
    assert result.state_weights is None
    cover = result.strict_cover
    assert cover.floor_surplus > 0
    assert (
        evaluate_partition_cover(yes, no, cover.yes_quantities, cover.no_quantities)
        == cover
    )


@pytest.mark.parametrize(
    "yes,no",
    [
        ([F(1, 3)] * 3, [F(2, 3)] * 3),
        ([None] * 3, [None] * 3),
        ([F(4, 5)] * 3, [None, F(1, 2), F(1, 2)]),
        ([F(2)] * 3, [F(2)] * 3),
    ],
)
def test_certificate_validates_available_sides_and_exact_unit_mass(yes, no):
    result = separate_partition_cover(yes, no)
    assert result.strict_cover is None
    validate_partition_cover_certificate(yes, no, result.state_weights)


def test_certificate_covers_all_small_mixed_unequal_portfolios():
    yes = [F(1, 5), F(2, 5), F(3, 5)]
    no = [F(9, 10), F(7, 10), F(1, 2)]
    result = separate_partition_cover(yes, no)
    validate_partition_cover_certificate(yes, no, result.state_weights)
    for quantities in product(map(F, range(3)), repeat=6):
        cover = evaluate_partition_cover(yes, no, quantities[:3], quantities[3:])
        assert cover.floor_surplus <= 0


def test_strictness_is_exact_at_far_sub_float_resolution():
    epsilon = F(1, 10**100)
    assert (
        separate_partition_cover(
            [F(1, 2), F(1, 2) - epsilon], [None, None]
        ).strict_cover.floor_surplus
        == epsilon
    )
    assert (
        separate_partition_cover([F(1, 2), F(1, 2)], [None, None]).strict_cover is None
    )


@pytest.mark.parametrize("bad", [True, 0.5, float("nan"), F(-1)])
def test_inexact_negative_or_boolean_costs_reject(bad):
    with pytest.raises(ValueError):
        separate_partition_cover([bad, F(1)], [None, None])


@pytest.mark.parametrize(
    "weights", [[], [F(0), F(0)], [F(-1), F(2)], [F(1), F(0)], [0.5, 0.5]]
)
def test_forged_or_non_covering_certificate_rejects(weights):
    with pytest.raises(ValueError):
        validate_partition_cover_certificate([F(1, 2)] * 2, [F(1, 2)] * 2, weights)


def test_unavailable_leg_is_never_a_free_position():
    with pytest.raises(ValueError):
        evaluate_partition_cover([None, F(1)], [None, None], [F(1), F(0)], [F(0), F(0)])


def test_invalid_dimensions_and_quantities_reject():
    for yes, no in (([F(1)], [F(1)]), ([F(1)] * 2, []), ([F(1)] * 4097, [F(1)] * 4097)):
        with pytest.raises(ValueError):
            separate_partition_cover(yes, no)
    for amounts in ([F(-1), F(0)], [True, F(0)], [F(0)]):
        with pytest.raises(ValueError):
            evaluate_partition_cover([F(1)] * 2, [F(1)] * 2, amounts, [F(0)] * 2)
