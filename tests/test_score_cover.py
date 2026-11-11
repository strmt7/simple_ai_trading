"""Independent payoff identities and exact certificate rejection controls."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from simple_ai_trading.score_cover import (
    ScoreCoverCertificate,
    ScoreMarket,
    representative_score,
    score_states,
    solve_score_cover,
    verify_score_certificate,
)


def markets():
    return (
        ScoreMarket("first", "team_total", 1, 0),
        ScoreMarket("second", "team_total", 1, 1),
        ScoreMarket("game", "total", 2),
    )


def test_reverse_additive_cover_is_an_exact_direction_independent_witness():
    system = score_states(markets())
    prices = (F(1, 5), F(19, 20), F(1, 5), F(19, 20), F(19, 20), F(1, 5))
    result = solve_score_cover(system, prices)
    verify_score_certificate(system, prices, result)
    assert result.cost_lower == result.cost_upper == F(3, 5)
    assert dict(result.primal) == {0: F(1), 2: F(1), 5: F(1)}
    # Independent score predicates, not the production payoff evaluator.
    for first, second in product(range(12), repeat=2):
        payoff = int(first >= 1) + int(second >= 1) + int(first + second < 2)
        assert payoff >= 1
    assert sum(dict(result.primal).values()) * F(1, 2) >= 1


def test_cancellation_dual_proves_rejection_of_all_fractional_portfolios():
    system = score_states(markets())
    result = solve_score_cover(system, (F(11, 20),) * 6)
    assert result.cost_lower == result.cost_upper == F(11, 10)
    verify_score_certificate(system, (F(11, 20),) * 6, result)


@pytest.mark.parametrize("scalar,margin", [(0, 1), (1, 1), (3, 2), (128, 64)])
def test_projection_preserves_all_rule_kinds_at_boundaries_and_arbitrary_large_scores(
    scalar, margin
):
    rules = [
        ScoreMarket("winner-a", "winner"),
        ScoreMarket("winner-b", "winner", team=1),
    ]
    if scalar:
        rules += [
            ScoreMarket("total", "total", scalar),
            ScoreMarket("team-a", "team_total", scalar),
            ScoreMarket("team-b", "team_total", scalar, 1),
        ]
    rules += [
        ScoreMarket("margin-a", "margin", margin),
        ScoreMarket("margin-b", "margin", margin, 1),
    ]
    bound = scalar + margin + 2
    values = {
        0,
        1,
        max(0, scalar - 1),
        scalar,
        scalar + 1,
        bound - 1,
        bound,
        bound + 1,
        10**50,
        10**50 + margin,
        10**50 + margin + 1,
    }
    for first, second in product(sorted(values), repeat=2):
        projected = representative_score(first, second, scalar, margin)
        assert min(projected) >= 0 and max(projected) <= bound
        for rule in rules:
            assert rule.first_payout_twice(first, second) == rule.first_payout_twice(
                *projected
            )


def test_winner_tie_and_common_void_pay_half_and_complements_reconcile():
    rules = (ScoreMarket("winner", "winner"), ScoreMarket("margin", "margin", 2))
    system = score_states(rules)
    assert rules[0].first_payout_twice(50, 50) == 1
    assert rules[0].first_payout_twice(51, 50) == 2
    assert rules[0].first_payout_twice(50, 51) == 0
    assert bytes([1] * 4) in system.rows
    for row in system.rows:
        assert all(row[index] + row[index + 1] == 2 for index in range(0, 4, 2))


@pytest.mark.parametrize(
    "mutation",
    [
        "negative",
        "missing-state",
        "wrong-hash",
        "overprice",
        "wrong-cost",
        "absent-price",
        "duplicate",
    ],
)
def test_certificate_tampering_rejects_without_numerical_tolerances(mutation):
    system = score_states(markets())
    prices = (F(11, 20),) * 6
    result = solve_score_cover(system, prices)
    if mutation == "negative":
        result = replace(result, primal=((0, F(-1)),))
    elif mutation == "missing-state":
        system = replace(
            system, rows=system.rows[:-1], representatives=system.representatives[:-1]
        )
    elif mutation == "wrong-hash":
        system = replace(system, matrix_sha256="0" * 64)
    elif mutation == "overprice":
        result = replace(
            result, dual=tuple((index, value * 2) for index, value in result.dual)
        )
    elif mutation == "wrong-cost":
        result = replace(result, cost_upper=result.cost_upper - F(1, 10**12))
    elif mutation == "absent-price":
        changed = list(prices)
        changed[result.primal[0][0]] = None
        prices = tuple(changed)
    else:
        result = replace(result, primal=result.primal + result.primal[:1])
    with pytest.raises(ValueError):
        verify_score_certificate(system, prices, result)


def test_missing_price_population_is_distinct_from_complete_rejection():
    system = score_states(markets())
    prices = (F(11, 20), F(11, 20), None, None, None, None)
    result = solve_score_cover(system, prices)
    assert result.complete_prices is False
    assert result.cost_lower == result.cost_upper == F(11, 10)
    verify_score_certificate(system, prices, result)


@pytest.mark.parametrize(
    "changes",
    [
        {"threshold": True},
        {"threshold": 129},
        {"kind": "unknown"},
        {"team": True},
        {"kind": "margin", "threshold": 65},
        {"kind": "winner", "threshold": 1},
    ],
)
def test_invalid_market_rules_fail_before_state_enumeration(changes):
    selected = dict(identifier="invalid", kind="total", threshold=1)
    selected.update(changes)
    with pytest.raises(ValueError):
        ScoreMarket(**selected)


def test_verifier_is_independent_of_scipy_and_accepts_a_manual_exact_proof(monkeypatch):
    system = score_states((ScoreMarket("binary", "total", 1),))
    cancellation = system.rows.index(bytes([1, 1]))
    proof = ScoreCoverCertificate(
        ((0, F(1)), (1, F(1))), ((cancellation, F(11, 10)),), F(11, 10), F(11, 10), True
    )
    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize, "linprog", lambda *a, **k: pytest.fail("verifier called solver")
    )
    verify_score_certificate(system, (F(11, 20), F(11, 20)), proof)
