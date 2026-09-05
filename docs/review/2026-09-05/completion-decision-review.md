# Completion must beat a feasible liquidation alternative

The new `src/simple_ai_trading/completion_economics.py` implements the missing
research comparator identified in the September 4 inventory/maker reviews.
It is not wired into trading, historical labels or model promotion. Its
[reproducible synthetic report](completion-decision-review.json) binds the
implementation, generator and focused tests. No historical outcomes changed.

Given existing net original shares `Q` and net acquired opposite shares `X`,
matched quantity is `min(Q,X)`. Both residual quantities are valued separately.
Completion value is matched-pair net realization plus residual net liquidation
value, less acquisition cash and additional costs. Incremental value subtracts
the executable full-quantity original-inventory liquidation comparator.
All inputs must share a condition/payoff identity, quote unit and valuation
horizon; the helper cannot certify those external premises or depth itself.

Missing prices for nonzero quantities reject calculation. Zero residuals do
not need fictitious bids. Net liquidation can be negative when costs exceed
proceeds. Acquisition cash already includes quote fees; net share quantities
already deduct share fees; costs may not be counted twice. Original acquisition
cost affects accounting PnL, but cannot change this incremental ranking.

Four deliberately synthetic examples, not current quotes or estimated fills:

| Example | Incremental value over liquidation | Worst supplied state |
| --- | ---: | ---: |
| Pair earns historical profit, but immediate sale is better | -5 | -5 |
| Partial completion with original residual | -11 | -11 |
| 90/10 illustrative full/partial mixture | +3.40 | -11 |
| Opposite-side overfill with residual disposal | -6 | -6 |

The mixture's weights are supplied assumptions, not calibrated probabilities.
Its positive mean is not profit in every state. Even all-positive supplied
states cannot prove the population is exhaustive, outcomes causal, prices
executable or net profitability stable. `qualified_edge` remains false.

## Consequence for the next model

The economic target is completion value minus feasible liquidation value,
not whether a completed historical pair had positive PnL. A useful model must
estimate joint completion/residual outcomes using causal features, with
partial/no-fill/failed cases and a held-out liquidation comparator. Independent
side probabilities cannot be multiplied without supporting evidence. Do not
train on the old 66 selected lock rows: their reviewed schema lacks that bid,
depth and fee comparator. No inferred zero comparator or retrospective future
hedge fill may repair the missing decision-time evidence.

The pure evaluator supports prospective specifications without a new data
request. Actual adoption still needs condition/quantity/source-clock binding,
execution feasibility, independently identified inventory, net fees and
untouched joint-outcome evidence. It cannot stand in for an event ledger,
collector, probability estimator, operational risk gate or enterprise readiness.

Twenty-four focused tests passed. The initial empty-state test exposed a test
helper defaulting an empty tuple to a full-fill fixture; that helper was fixed
and the rejection branch now runs. No broad CI, training or GPU timing was run.

## Holding-yield route checked, not reopened

One prospectively recorded search returned unrelated event pages, not program
terms. This is inadequate retrieval, not proof that the program is unchanged.
No material-change trigger was established; V7 partial data, consumed pulses,
receipts and bridge quotes were untouched. Incidental event prices were excluded
from evaluation and prospective promotion; no event was followed or screened.
The full search extraction and plan are bound by the report. Do not repeat this
ineffective query. Canonical counts and every market retry trigger stay unchanged.
