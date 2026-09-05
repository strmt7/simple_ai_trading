# Forward make/take evaluation: prevent role reuse and missing-day claims

The reusable Round 57 economic evaluator can pass a synthetic evaluation
whose declared six days overlap its two policy-calibration days, although its
target path maps cover only those two days. The unchanged existing fixture
produces 33 closed trades and a passing economic gate. This is an API contract
gap, not an observed profitable strategy or proof of historical experiment
leakage. The current frozen runner's `load_round57_contract` independently
checks contiguous, chronological role windows; its implementation and prior
reports are preserved.

The new `make_take_forward_evaluation.evaluate_make_take_policy_forward`
wraps the old evaluator for future work. It requires complete supplied
calibration/evaluation batches, consecutive integer two/six-day roles, and
evaluation days strictly later than calibration days. It validates every
symbol's recorded target day-path coverage and checks every decision lies in
its declared role. Calibration action/target hashes must match the selected
policy's complete source bindings, not a newly substituted convenient subset.

The last recorded valid calibration label across **all** base/stress target
rows must precede the earliest evaluation decision, even when that calibration
row was not selected as an order. Equality fails. This blocks outcome leakage
through label horizons that a date-only split would miss. It returns the old
economic result inside a source-linked chronology record with its own hash;
`qualified_edge` stays false. Session instructions route new evaluations here,
without changing or replaying the frozen Round 57 runner.

## Reproduced evidence

The [diagnostic JSON](make-take-forward-boundary.json) binds twelve source/test
files and records the synthetic legacy counterexample. A separately constructed
control covers calibration days 0-1 and evaluation days 2-7 with all required
day-path hashes. Its 99 synthetic closed trades pass the unchanged economic
gate and the new chronology boundary. Neither test contains actual market
profit or a trained model improvement.

The new regression suite checks the legacy gap, forward control, date/type
failures, source substitution, out-of-role decisions, and calibration labels
ending exactly at, just after and just before the first evaluation decision.
The initial shared calibration fixture failed because it moved rows onto day 1
without adding that day's path hash. Only the new synthetic fixture was given
an explicit two-day map and recalibrated; the old fixture was not rewritten,
and the new coverage guard was not weakened.

Nineteen focused checks passed across the new forward/diagnostic cases and
the unchanged action-value, policy and economic-evaluation tests. Ruff passed;
no broad CI, model fit or performance benchmark was run.

## Limits and next action

This guard proves relationships among supplied, hash-validated batches. A
day-path hash is not proof of source-file availability or continuous raw
capture, and the wrapper does not independently recompute predictive metrics,
certify model-training separation, correct partial-fill/no-full-fill PnL, or
establish a probability of future profitability. Those remain separate gates.
Sparse days can have no selected trades; they still need source day-path
evidence before being counted as covered evaluation days.

No source-bound historical implementation, result, population, market retry
trigger or acceptance count changed. No market request, GPU training, account
or order access occurred. The earlier recent-literature discovery was reused
for routing and not repeated. The literature source output initially hit the
Windows console's legacy text encoding; printing ASCII-escaped retained JSON
resolved that local display error without reaccessing any source.

Next forward make/take work must pass this boundary before treating a terminal
score as held-out evidence, while continuing the existing quantity-aware
partial-fill and executable-cost requirements. Do not retrain on consumed
data to turn this synthetic validation fix into a profitability claim.
