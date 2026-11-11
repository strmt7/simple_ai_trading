# Joint score covers and the residual-payoff boundary

The retained Bills–Vikings response admits an exact one-unit-cost cover under
the declared common integer-score model. The minimum is **1**, and the payoff
floor is **1**. No strictly sub-unit long-only fractional cover exists among
the 156 priced outcomes in that retained proxy system. Expected profit remains
unproved. This is retrospective exploration of consumed public bytes, with no
new observation, quote, fee, account request or order.

The original [November 10 population](../../2026-11-09/nfl-nov10-total-floor/result.json)
tested all 210 two-total pairs and rejected their gross cover costs. It remains
unchanged. This [separately frozen contract](contract.json) asks a broader
allocation question on the original selected event: every nonnegative fractional
portfolio of its 78 full-game moneyline, spread, team-total and game-total markets.
Period markets, props, shorts, dynamic trading, rewards and financing are outside
this calculation. The old selection history is retained; no untouched holdout
or independent recurrence is restored.

## Complete conditional payoff model

All 78 complete rule templates passed before their new joint economic fields
were consulted. Outcomes, favored teams, thresholds, overtime clauses where
specified, postponement, common whole-game cancellation, condition IDs and tokens
are bound to the exact original response. Their recorded resolution-source URLs
are retained. Two final nonnegative integer scores determine the admitted
payoffs; an actual moneyline tie pays one half. Common cancellation pays one
half per outcome. Independent disputes, partial voids, actual oracle/collateral
identity and the interpretation of full-game score references remain separate
qualification requirements. The floor is conditional on this rule model.

There is no assumed empirical maximum score. Let `T` be the largest team/game
total threshold and `D` the largest absolute margin threshold, at least one.
Every score pair has a payoff-equivalent representative in the square
`[0, T + D + 2]²`. If the smaller score is at most `T`, clamp a larger score
to that bound: its lead remains above `D`. If both exceed `T`, set their minimum
to `T + 1` and preserve their difference clipped to `[-D - 1, D + 1]`. All
admitted threshold, margin and tie predicates survive. Enumerating that finite
square and common cancellation therefore covers the whole declared score domain.

Here `T = 65`, `D = 22` and the bound is 89. Deduplication before prices leaves
1,171 distinct payoff vectors. This conservative integer domain can include
unreachable sporting scores; the negative result applies to the stated model,
not every possible refinement of sporting rules or every trading strategy.

The optimization minimizes retained proxy cost subject to at least one payoff
unit in every state. First outcomes use `bestAsk`; second outcomes use
`1 - bestBid` as the existing rejection proxy. `outcomePrices` are never used.
All 156 selected-side proxies are present. Zero acquisition quotes remain
missing inventory. These fields prove no current executable ask, capacity,
synchronization, fill or commission rate.

## Exact certificate

The numerical solver proposes weights once. Fixed rational reconstruction scales
the portfolio to its exact worst-state floor, and scales nonnegative dual weights
down until every priced-leg inequality holds. Every inequality is then checked
with fractions, without a numerical acceptance tolerance.

The [result](result.json) supplies both sides of the bound. A feasible portfolio
costs exactly one unit. A 54-state nonnegative dual witness sums to exactly one
and fits beneath every retained leg price, proving the matching lower bound on
all feasible nonnegative fractional portfolios in this proxy system. The dual
weights are algebraic pricing witnesses; they are not forecast probabilities or
training labels. A numerical solution with unequal certified bounds would remain
an interval rather than a claim of exact optimality.

The standalone [rational audit](reconcile.py) imports neither NumPy nor SciPy.
It independently reconstructs native line/team mappings, quotes, the complete
payoff matrix, all portfolio-state inequalities and all dual-leg inequalities.
It confirms minimum payoff 1, maximum payoff 2 and both objective bounds 1.
Resealed mutations to a threshold, the objective and trading authority reject.
[Reconciliation receipt](independent-reconciliation.json)

## Expected-value question left open

The returned minimum-cost witness holds one Vikings-minus-8.5 first outcome
at the retained 0.15 proxy and one Vikings-minus-9.5 second outcome at the
retained 0.85 proxy. Under the admitted common rules its terminal payoff is
`1 + 1{Vikings final margin = 9}`; common cancellation pays 1.

At a common cash horizon, let `C` include every acquisition premium over one,
commission, execution loss, financing, opportunity cost, settlement delay and
other incremental cost. The residual claim requires independently supported
probability of a nine-point margin to exceed expected `C`, together with a
separate allowance for inconsistent settlement and other failures outside the
model. No probability, native cost or positive residual value is assigned here.
The historical study's unrealized middle payout receives no credit.

This solver selected a minimum-cost witness; it did not rank expected value.
Any later expected-value study needs a complete candidate/selection record,
independent chronological score-probability calibration, tail bounds, actual
matched prices and costs, and a feasible idle-cash comparator. The consumed
snapshot cannot become a holdout, a positive edge, or permission to obtain books
by adding a probability assumption. Its exact retry boundaries remain unchanged.

## Verification and scope

Before economic access, 34 focused controls passed. They cover arbitrary large
score projection, ties and cancellation, a hand-proved synthetic reverse-additive
cover, exact primal/dual bounds, absent prices, forged/missing states, malformed
rules, native team orientation and the real one-use entry point. The first
price-adapter control exposed a zero-ask ordering defect: zero must become
missing before checking for crossed positive quotes. It was repaired and the
failed pre-economic phase is preserved. No historical financial result changed.

The actual calculation ran once in a hidden owned child with a 120-second outer
deadline and a 20-second solver limit. Network access and further subprocess
creation were denied in that child. Only its own CPU affinity was limited to
two available logical processors; other tasks were untouched. The small static
program required no GPU fit, package installation, service or scheduler.

[Sealed evidence](evidence.json) binds the source phases, search receipts, controls,
raw log hashes, exact result and independent audit. Registry counts remain
206 observations, 65 hypotheses, 37 historical scoped mechanisms and zero
qualified stable edges. The whole product revamp and final review remain active.
