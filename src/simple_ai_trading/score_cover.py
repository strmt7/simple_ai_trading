"""Exact certificates for conditional integer-score portfolio payoff covers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from typing import Literal


@dataclass(frozen=True)
class ScoreMarket:
    """First-outcome rule; the second outcome is its exact complement.

    Scores are final nonnegative integers for the same two participants. Common
    whole-event cancellation pays one half on each outcome. Independent disputes,
    partial voids, different observation scopes and actual costs are not modeled.
    """

    identifier: str
    kind: Literal["team_total", "total", "margin", "winner"]
    threshold: int = 0
    team: int = 0

    def __post_init__(self):
        if (
            not isinstance(self.identifier, str)
            or not 0 < len(self.identifier) <= 128
            or self.kind not in {"team_total", "total", "margin", "winner"}
            or type(self.team) is not int
            or self.team not in {0, 1}
            or type(self.threshold) is not int
            or self.kind == "winner"
            and self.threshold != 0
            or self.kind != "winner"
            and not 1 <= self.threshold <= (64 if self.kind == "margin" else 128)
        ):
            raise ValueError("score market exceeds its exact rule or resource scope")

    def first_payout_twice(self, first: int, second: int) -> int:
        """Use doubled units to preserve half payouts without floating point."""
        if type(first) is not int or type(second) is not int or min(first, second) < 0:
            raise ValueError("scores must be nonnegative integers")
        own, other = (first, second) if self.team == 0 else (second, first)
        if self.kind == "winner":
            return 1 if own == other else 2 * int(own > other)
        value = (
            first + second
            if self.kind == "total"
            else own - other
            if self.kind == "margin"
            else own
        )
        return 2 * int(value >= self.threshold)


def representative_score(
    first: int, second: int, scalar_limit: int, margin_limit: int
) -> tuple[int, int]:
    """Map every score to a finite square while preserving every admitted rule.

    Set B=T+D+2, where scalar thresholds are at most T and absolute margin
    thresholds are at most D. If the smaller score is at most T, clamping the
    larger to B preserves every scalar predicate and leaves its lead above D.
    Otherwise both scores exceed all scalar thresholds: set their minimum to
    T+1 and preserve their difference, clipped to [-D-1,D+1]. Winner ties and
    every margin predicate are preserved. No maximum real-world score is assumed.
    """
    if (
        type(first) is not int
        or type(second) is not int
        or min(first, second) < 0
        or type(scalar_limit) is not int
        or not 0 <= scalar_limit <= 128
        or type(margin_limit) is not int
        or not 1 <= margin_limit <= 64
    ):
        raise ValueError("score projection scope is invalid")
    bound = scalar_limit + margin_limit + 2
    if max(first, second) <= bound:
        return first, second
    if min(first, second) <= scalar_limit:
        return min(first, bound), min(second, bound)
    difference = max(-margin_limit - 1, min(margin_limit + 1, first - second))
    lower = scalar_limit + 1
    return lower + max(0, difference), lower + max(0, -difference)


@dataclass(frozen=True)
class ScoreStateSystem:
    markets: tuple[ScoreMarket, ...]
    rows: tuple[bytes, ...]
    representatives: tuple[tuple[int, int] | None, ...]
    scalar_limit: int
    margin_limit: int
    bound: int
    matrix_sha256: str


def score_states(markets: tuple[ScoreMarket, ...]) -> ScoreStateSystem:
    """Enumerate all distinct score payoffs plus common cancellation before prices."""
    if (
        not isinstance(markets, tuple)
        or not 1 <= len(markets) <= 160
        or any(not isinstance(market, ScoreMarket) for market in markets)
        or len({market.identifier for market in markets}) != len(markets)
    ):
        raise ValueError("score population is missing, duplicated or oversized")
    scalar = max(
        (m.threshold for m in markets if m.kind in {"team_total", "total"}), default=0
    )
    margin = max((m.threshold for m in markets if m.kind == "margin"), default=1)
    bound = scalar + margin + 2
    unique: dict[bytes, tuple[int, int] | None] = {}
    for first in range(bound + 1):
        for second in range(bound + 1):
            row = bytearray()
            for market in markets:
                payout = market.first_payout_twice(first, second)
                row.extend((payout, 2 - payout))
            unique.setdefault(bytes(row), (first, second))
    unique.setdefault(bytes([1] * (2 * len(markets))), None)
    rows = tuple(sorted(unique))
    return ScoreStateSystem(
        markets,
        rows,
        tuple(unique[row] for row in rows),
        scalar,
        margin,
        bound,
        hashlib.sha256(b"".join(rows)).hexdigest(),
    )


@dataclass(frozen=True)
class ScoreCoverCertificate:
    """Primal and dual witnesses; all quantities and costs remain exact."""

    primal: tuple[tuple[int, Fraction], ...]
    dual: tuple[tuple[int, Fraction], ...]
    cost_upper: Fraction
    cost_lower: Fraction
    complete_prices: bool


def _prices(
    system: ScoreStateSystem, prices: tuple[Fraction | None, ...]
) -> tuple[int, ...]:
    if not isinstance(system, ScoreStateSystem) or system != score_states(
        system.markets
    ):
        raise ValueError("score state system differs from its complete rule population")
    if (
        not isinstance(prices, tuple)
        or len(prices) != 2 * len(system.markets)
        or any(
            p is not None
            and (
                not isinstance(p, Fraction)
                or max(p.numerator.bit_length(), p.denominator.bit_length()) > 4096
                or not 0 <= p <= 1
            )
            for p in prices
        )
    ):
        raise ValueError(
            "score prices require exact unit fractions or explicit absence"
        )
    return tuple(index for index, value in enumerate(prices) if value is not None)


def _weights(
    values: tuple[tuple[int, Fraction], ...], limit: int
) -> dict[int, Fraction]:
    if not isinstance(values, tuple) or len(values) > limit:
        raise ValueError("score witness is oversized")
    result = {}
    for index, weight in values:
        if (
            type(index) is not int
            or not 0 <= index < limit
            or index in result
            or not isinstance(weight, Fraction)
            or weight <= 0
            or max(weight.numerator.bit_length(), weight.denominator.bit_length())
            > 4096
        ):
            raise ValueError("score witness has an invalid index or weight")
        result[index] = weight
    return result


def verify_score_certificate(
    system: ScoreStateSystem,
    prices: tuple[Fraction | None, ...],
    certificate: ScoreCoverCertificate,
) -> None:
    """Verify both bounds with rational arithmetic, without the numerical solver.

    For every state A*w>=1 proves the supplied portfolio's unit floor. For every
    priced leg A.T*y<=price with y>=0 proves a lower bound on every feasible
    nonnegative portfolio using those legs. Missing legs are never priced at zero.
    """
    priced = _prices(system, prices)
    if not isinstance(certificate, ScoreCoverCertificate):
        raise ValueError("score certificate type differs")
    primal = _weights(certificate.primal, len(prices))
    dual = _weights(certificate.dual, len(system.rows))
    if any(index not in priced for index in primal):
        raise ValueError("score witness spends an unpriced leg")
    for row in system.rows:
        if (
            sum((weight * row[index] for index, weight in primal.items()), Fraction())
            < 2
        ):
            raise ValueError("score portfolio does not cover every state")
    for column in priced:
        load = (
            sum(
                (weight * system.rows[index][column] for index, weight in dual.items()),
                Fraction(),
            )
            / 2
        )
        if load > prices[column]:
            raise ValueError("score lower-bound witness exceeds a leg price")
    upper = sum(
        (prices[index] * weight for index, weight in primal.items()), Fraction()
    )
    lower = sum(dual.values(), Fraction())
    if (
        type(certificate.complete_prices) is not bool
        or certificate.complete_prices != (len(priced) == len(prices))
        or not isinstance(certificate.cost_upper, Fraction)
        or not isinstance(certificate.cost_lower, Fraction)
        or certificate.cost_upper != upper
        or certificate.cost_lower != lower
        or lower > upper
    ):
        raise ValueError("score certificate costs or population do not reconcile")


def solve_score_cover(
    system: ScoreStateSystem,
    prices: tuple[Fraction | None, ...],
) -> ScoreCoverCertificate:
    """Find a numerical candidate once, then certify exact conservative bounds.

    Solver success is insufficient. Fixed one-million-denominator reconstruction
    scales the primal to its exact worst-state floor and scales the nonnegative
    dual downward until every priced-leg constraint holds. No tolerance accepts
    a financial inequality. The resulting interval need not prove optimality.
    """
    import numpy as np
    from scipy.optimize import linprog

    priced = _prices(system, prices)
    if not priced:
        raise ValueError("no priced score legs")
    matrix = (
        np.frombuffer(b"".join(system.rows), dtype=np.uint8)
        .reshape(len(system.rows), len(prices))[:, priced]
        .astype(float)
        / 2
    )
    result = linprog(
        np.array([float(prices[index]) for index in priced]),
        A_ub=-matrix,
        b_ub=-np.ones(len(system.rows)),
        bounds=(0, None),
        method="highs-ds",
        options={"time_limit": 20.0},
    )
    if not result.success:
        raise ValueError("score solver did not produce an admitted candidate; no retry")

    def rational(value):
        if not math.isfinite(float(value)):
            raise ValueError("score solver returned a nonfinite witness")
        return Fraction(str(max(0.0, float(value)))).limit_denominator(1_000_000)

    primal = {
        column: value
        for column, raw in zip(priced, result.x, strict=True)
        if (value := rational(raw)) > 0
    }
    minimum = min(
        sum((weight * row[index] for index, weight in primal.items()), Fraction()) / 2
        for row in system.rows
    )
    if minimum <= 0:
        raise ValueError("score reconstruction has no full-quantity floor")
    primal = {index: weight / minimum for index, weight in primal.items()}
    dual = {
        index: value
        for index, raw in enumerate(-result.ineqlin.marginals)
        if (value := rational(raw)) > 0
    }
    scale = Fraction(1)
    for column in priced:
        load = (
            sum(
                (weight * system.rows[index][column] for index, weight in dual.items()),
                Fraction(),
            )
            / 2
        )
        if load > 0:
            scale = min(scale, prices[column] / load)
    dual = {
        index: weight * scale for index, weight in dual.items() if weight * scale > 0
    }
    certificate = ScoreCoverCertificate(
        tuple(sorted(primal.items())),
        tuple(sorted(dual.items())),
        sum((prices[index] * weight for index, weight in primal.items()), Fraction()),
        sum(dual.values(), Fraction()),
        len(priced) == len(prices),
    )
    verify_score_certificate(system, prices, certificate)
    return certificate
