# Search bookkeeping cannot manufacture model evidence

The training suite's preliminary selection guard could change an unchanged
selected model from fail to pass by adding copies of losing candidates. Two
separately specified synthetic reproductions demonstrate distinct mechanisms:

| Retained reproduction | Original result | Repaired result |
| --- | --- | --- |
| Selected score 0.00005; pool sizes 3, 30, 300, 3,000 | Fail, fail, pass, pass | All fail; penalties increase |
| Reversed selection/validation rankings; pool sizes 4, 7, 40 | Fail, pass, pass; rank proxy changes from 1 to 0 | All fail; rank proxy remains 1 |

These are bookkeeping counterexamples, not financial returns or new market
observations. Copies add no independent return history. The original numerical
results, prospective synthetic inputs and exact source phase are retained.

## Corrected mechanism

The score guard no longer divides its penalty by the square root of the number
of candidate scores. That count measures search breadth, not the selected
strategy's time-series sample length. Admission now uses the finite score range;
IQR is diagnostic only, because copying near-identical scores can shrink IQR.
The existing fixed/relative floors and logarithmic trial multiplier are retained.

For fixed finite selected score `x`, let `R` be the candidate score range and
`N` the counted trials. The guard uses:

`penalty = log(N + 1) * max(R, 0.03 * abs(x), 0.0001)`

If candidates are added without decreasing declared search counters, `N` and
`R` cannot decrease. Thus this penalty cannot decrease. This is an algebraic
admission invariant, not a fitted uncertainty estimate. It does not derive
statistical confidence from the range, and these constants are not calibrated
significance levels. A valid model can still fail this preliminary guard.

The two-panel rank proxy counts each exact finite selection/validation score
pair once. Supplied counts and candidate copies remain in the score penalty; deduplicating
rank profiles is not an independence adjustment or a reduction in search budget.
Tied panel winners use the worst opposite-panel score, preserving input-order
invariance. An equivalent selected profile retains its diagnostic rank even if
it is represented by another dictionary object. Missing comparable splits fail.

Boolean and unrepresentable scores are not numeric evidence. Explicit search
and variant counts must be integers in their valid ranges; malformed counts
cannot be truncated, coerced or silently replaced by one. Large valid integer
counts retain their exact search breadth without a float conversion. Opposite-sign
finite quantile endpoints no longer overflow during interpolation. Overflowing
range or panel differences reject with JSON-safe unavailable fields. Proxy
admission requires explicit Boolean true rather than a truthy status.

New method identifiers and `statistical_significance_qualified: false` accompany
the reports. Legacy `deflated_score`, `effective_trials` and
`probability_backtest_overfit` fields remain for compatibility, with explicit
method/interpretation metadata. They are not DSR, calibrated PBO, or a probability
that a model will make money. Historical model/result bytes remain unchanged.
Missing legacy internal-variant metadata retains a one-trial compatibility
interpretation; complete historical search provenance is still unproved.

## Verification

All **124** checks in the final selection-integrity, training-suite and model-lab
lane pass. Six regression cases fail against the original training module at
the incorrect numeric/rank/pass behavior, without collection errors. The private
controller restores exact working bytes even on failure. Preceding 104/108-case
lanes predate the numeric-input/count refinements and remain separate source
phases with their original evidence retained. Counts must not be summed.
One existing synthetic expectation now rejects the wide, 2,023-trial search;
no production rule was weakened to preserve its previous pass.

Semantic review and AST comparison isolate four changed production definitions:
numeric admission, quantile interpolation, rank diagnostics and the score guard.
No model fit, objective, feature, terminal-holdout ledger or training/backend
implementation changed. Bounded tests may fit synthetic fixtures; no production
model was retrained and no new performance claim follows. The
[evidence](evidence.json) binds source snapshots, receipts and the actual phases.

## Statistical and financial work still required

The [primary-source review](sources.md) admits current author publication metadata
and preserves three failed article captures. It does not qualify unseen formulas.
A proper next inference implementation needs complete source methods, objective-
appropriate net return paths at matched observation intervals, dependence and
sample-length assumptions, all tuning/search trials, and untouched evaluation
with repeated-access controls. Synthetic calibration and adversarial checks must
precede financial evaluation. Do not substitute candidate counts for temporal
observations, claim independence from different seeds, or retune a consumed test.

Native cash/clock/cost coverage and equal-capital benchmarks remain prerequisites
for expanded financial fits. Statistical inference alone cannot establish current
fees, fills, collateral, borrowing, custody or executable capacity. No market,
account, credential, order or protected-capture access occurred. Counts remain
**204 observations / 65 hypotheses / 37 scoped mechanisms / 0 qualified stable edges**.
