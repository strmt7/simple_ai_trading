# Model evidence: repeatable economics before model complexity

Review base: `9b43812aa3a1ac440bf78bfde0af24f8fadd9113`.
This is a source review and synthetic counterexample study, not a fresh market
experiment. No protected capture or historical outcome was reopened or edited.

## Confirmed promotion defect and correction

`model_lab._outcome_from_suite` filtered out absent selection-risk reports, then
accepted every surviving report unless `passed` was literally false. Empty
evidence, missing status, truthy strings and a passed report for only one of
several objectives could therefore pass this control. This is distinct from
proving that an actual published historical strategy exploited the defect.

The corrected boundary requires each objective's own explicit Boolean pass,
no failure reason and an empty rejection list. Malformed reports are recorded
as unavailable evidence rather than raising during dictionary conversion.
Positive infinity is rejected as a selected score or walk-forward performance
statistic. Twelve counterexamples failed on the old implementation; they pass
after the correction, alongside valid two-objective acceptance. Existing tests
of later stress, temporal, learning and portfolio controls now explicitly
supply valid selection-risk evidence so they continue testing their own gate.

Forty-six affected tests passed after the final dependency update, including
model-lab promotion, portfolio risk, foundation loading and dependency contracts.
This improves rejection correctness, not measured strategy profitability.

## Selection methodology limitation

The reviewed `training_suite._selection_risk_report` explicitly implements a
heuristic score haircut using candidate-score dispersion, a log trial-count
factor and the square root of the number of finite candidate scores. Its
`deflated_score` is not a Deflated Sharpe Ratio or a calibrated significance
probability. `_overfit_diagnostics` explicitly calls its two-panel calculation
a proxy, not full combinatorial validation. Preserve these historical fields
and outputs, but do not present them as institutional statistical certification.

The original [Deflated Sharpe Ratio paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
addresses selection bias and non-normal returns. The
[multiple-signal backtesting paper](https://www.nber.org/papers/w21329)
also makes search breadth material to inference. These methodological sources
motivate the review; neither establishes a Polymarket or Binance trading edge.

## Financially sound model-development direction

1. **Mechanism before fit.** Write the complete cash-flow identity and explain
   why the opportunity can recur. Separate deterministic matched-claim payoff
   from uncertain fills, funding, liquidation, financing and venue exposure.
2. **Predict the unresolved economic quantity.** For matched-position Binance
   carry, estimate integrated funding and exit-basis/cost risk, not merely the
   direction of price. For Polymarket completion, estimate fill/completion and
   adverse-selection costs conditional on legitimately preexisting inventory.
   Never manufacture the first risky leg to call the second leg arbitrage.
3. **Benchmark against simple decisions.** Compare with abstention, a fixed
   economic break-even rule and a small calibrated model on identical decision
   times, costs and capital. Complexity earns admission only through additional
   out-of-sample net value, not classification accuracy alone.
4. **Training owns tuning.** Fit transforms, features, thresholds, abstention,
   hyperparameters and costs only in assigned development roles. Split by
   decision/label interval and event, purge overlap, and preserve all searched
   families. More rows from one event are not independent replications.
5. **Measure uncertainty and concentration.** Use prospective, preregistered
   event/block-resampled paired net cash flows; report capital employed,
   time at risk, capacity, expected shortfall, drawdown and contribution by
   independent event/asset. A high win rate or reward/risk target is insufficient.
6. **Qualification is separate.** Preserve the cross-regime contract's
   abstention for unsupported regimes. A statistical diagnostic cannot supply
   current fees, borrow, eligibility, ownership or executable capacity.

The next economic study should develop a fixed break-even/abstention baseline
for the retained existing-inventory lock observations, with exploratory labels
and no reselection claim. Only after that rule and its independent evaluation
population are frozen should a learned completion-cost model be compared.
The Binance analogue must integrate the whole matched carry lifecycle and
total collateral requirement. Neither study has been run by this patch.

## Remaining coverage

Reviewed the model-lab promotion boundary, walk-forward numeric acceptance and
the training suite's selection-haircut/two-panel proxy implementation. Full
training data lineage, every objective, all model implementations, all UI paths
and every historical text remain broader review work; this document does not
claim they have been completed. Historical source-bound implementations and
outcomes remain preserved, with corrections recorded as new work.
