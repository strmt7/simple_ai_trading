# Stateful inventory cash and aggregation correction

Baseline: `d3d05ac59371191766beef5ecfc6fdeb955870d8`.
This is forward accounting implementation, not new market evidence. No network
market requests, model fit, inference, GPU benchmark, account operation or order.
The goal remains active; no automation was created or edited.

## Correction retained separately from historical results

The earlier round-trip diagnostic measured the **sum of the hourly return array**.
It did not trace `_economic_metrics`, which compounds that array. Interpreting
the sum as a reported +50% profit was unsupported. Its original test and receipt
remain unchanged; this adjudication supersedes only that interpretation.

For synthetic prices 100 -> 200 -> 100, all three equal sleeves long, no funding,
and a named 6-bp one-way cost model:

| Measure | Result | Meaning |
| --- | --- | --- |
| Legacy hourly array sum | +4,988 bps | Additive diagnostic, not total return |
| Legacy compounded report | -14.9964 bps | `(1 + .9994) * (1 - .5006) - 1` |
| Fixed one-unit cash on opening 100 quote | -12 bps | Gross zero, modeled trade cost 0.12 quote |

Zero-cost full-long common-price returns telescope under compounding. This
control does not prove the legacy report fabricates profit. What remains
different is the sizing/normalization convention: actual fixed units, traded
quote and fee cash are absent from the legacy normalized-label representation.
Constant original quote notional would instead require an interior -0.5-unit
resize at 200. The new route explicitly chooses fixed base quantity until a
direction change; it does not silently choose constant quote exposure.

## Implemented boundaries

- `funding_cash_inventory.py`: immutable exact Fraction paths and event/interval
  ledgers. Reuses the existing payment law. Charges absolute quantity change at
  each supplied boundary price and terminal close, including reversal notional.
  An unchanged holding across a shared boundary receives one payment. At an
  actual entry/change/exit boundary, old/new/flat extrema enclose uncertain
  entitlement; partial inventories lie between these endpoints. This assumes
  the named boundary execution model, not unbounded native fill delays.
- `stateful_position_policy.py`: existing strict turnover hurdle, reversal,
  maximum hold and terminal-close decisions, shared by both replay routes.
  No future target is read to produce decisions.
- `stateful_cash_replay.py`: explicit dataset/forecast grid, BTC/ETH/SOL prices,
  marked funding and population/coverage binding, starting capital and cost
  fraction. No missing-price interpolation or proxy fallback. Prices preserve
  the supplied binary float values, not reconstructed native decimal bytes.
  Exact interval cash is added to starting equity; total return divides total
  cash by that explicit reference. Replay identity binds actual boundary prices,
  forecasts, policy, costs, capital and marked-source provenance.

Source digests identify supplied evidence; they do not authenticate origin or
prove independent population completeness. Costs are a caller-supplied aggregate
sensitivity, not measured fees/spread/slippage. Perpetual price P&L is marked
interval cash in this conditional model; actual account settlement/fees require
native reconciliation. Boundary equity exhaustion is flagged, not prevented:
no intrabar liquidation, financing, leverage or automatic capital protection is
proved. The ledger can continue diagnostically after exhaustion, never execute.

Four of five audited **training builders** have supplied-cash paths. This new
stateful **replay** does not repair its scalar training targets, fit any model,
or provide a direction-neutral hedge. Side-specific adverse labels and matched
hedge quantities/costs/margin still precede affected training and admission.

## Verification and next decision

225 affected unit checks pass, including 62 new cases. Ruff passes for all five
touched Python files. One zero-network comparison loads the exact committed
baseline in memory and compares every replay array plus all metrics for both
modes on four 72-hour controls: persistent 25-bp signal, flat, strict-boundary
discrete signals and Gaussian signals. Targets use `default_rng(4301)` normal
values (mean 1, standard deviation 20); discrete/Gaussian forecasts follow from
that same RNG. All eight comparisons are exact, including final-cost arithmetic.
These are synthetic regression controls, not financial observations or timing.

The source-bound implementation receipt records final file identities and
commands. An initial new-test decorator typo was caught before collection,
corrected, then formatting/lint and the complete affected lane passed. No
historical test, receipt, raw capture, market registry or financial ledger was
edited. Counts remain 202 observations, 65 hypotheses, 37 mechanisms, zero
qualified stable edges.

Next: complete source-qualified direction-neutral hedge cash and financing/
margin/execution budgets, reusing retained native source populations only under
their existing capture boundaries. Then design two-sided surplus labels and
paired causal model objectives; do not retrain rate proxies or reopen consumed
screens merely because accounting controls pass. Trace final metric aggregation
before making performance claims; that requirement is now in `AGENTS.md`.
