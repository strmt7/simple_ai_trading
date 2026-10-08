# Hourly two-sided supplied cash labels

The forward `build_stateful_cash_labels` adapter retains separate long/short
adverse and upper cash labels. It does not rewrite Round 43's signed targets,
models or results, and it cannot enter the legacy signed trainer. The original
builder remains a rate proxy. Four existing builders have supplied-cash modes;
this is a separate adapter for the fifth consumer, not five admitted datasets.

## Economic question and result

A signed target `(long_net - short_net) / 2` is a directional contrast, not
either action's net cash. Equal execution costs cancel. Under uncertain funding
entitlement, the two adverse bounds also need not be opposites. That contrast
cannot alone serve as a two-sided surplus/abstention objective. This is not a
claim that the old cost-threshold policy ignored all fees or that a passing
accounting control supplies an edge.

At a supplied entry price of 100, a 0.005 funding rate and settlement mark of
102 produce 51 bps of fixed-base payment, not 50. With modeled 6-bp one-way
cost and unchanged exit price, interior held payment gives long -63 bps and
short +39 bps. An entry-boundary positive payment and exit-boundary negative
payment, each with unknown entitlement, instead give **both** sides an adverse
-63-bp bound and +39-bp upper bound. Their adverse signed contrast is zero;
neither bound may be obtained by negating the other side's adverse bound.

At exit price 200, the same fixed-base quantity costs 6 bps to enter and
12 bps to close, normalized by entry notional: 18, not a constant 12 bps.
The new adapter retains exact supplied float-price fractions for this price
and cost calculation, then combines outward-enclosed funding cash. It retains
float64 label bounds and rejects overflow, missing boundaries, nonfinite data,
misbound symbols, conflicting populations, missing cash inputs and scope gaps.

All controls are synthetic, not market observations. Eighteen combinations of
signed rate, boundary entitlement and exit price compare both label sides to
the exact independent-row inventory replay. They share the funding law, so
this verifies integration/enclosure, not an independent exchange-law proof.
Legacy target arrays never enter labels or feature bindings. Tests mutate those
targets without changing the new result, and change actual features/prices/marks
or their source identity to verify binding sensitivity.

## Scope and admission

Each row models an independently opened and closed one-hour fixed-base position
at supplied opens one minute after decision and 60 minutes after entry. It is
not a continuous stateful holding policy, a constant-quote rebalance, complete
spot/perpetual hedge, actual fee schedule or observed fill. The existing
`replay_stateful_fixed_base_cash` remains the continuous inventory adapter.
Neither path establishes funding eligibility from actual fills, native origin,
price/mark coverage, basis, financing, capital, margin or liquidation safety.

The new batch binds the actual feature/clock grid, selected price values,
supplied funding provenance, exact cost fraction and all output arrays.
Digests are lineage, not authenticity. Frozen dataclasses and read-only array
flags are accidental-mutation guards, not hostile-process security.
`financially_qualified` is always false. The legacy trainer rejects this batch
before creating a model directory. No model fit, inference or GPU benchmark ran.

Before training, qualify the full native paired cash/cost population and causal
feature availability, freeze a two-sided action/abstention objective with
stateful transitions, and bind chronological selection/holdout roles. Reuse
existing paired/payoff tooling; do not reinterpret the signed trainer as repaired
or expand a fit on old missing marks. No new funding-cash dataset is admitted.

## Verification and preservation

[Implementation receipt](stateful-two-sided-cash-implementation.json) binds
source/tests, exact synthetic controls and prior unchanged receipts. The affected
lane passed 241 checks, including 50 new cases; Ruff passed on three touched
Python files. Full pytest, coverage, training and timing benchmarks were not run.
Initial tests had an incorrect positional FundingState fixture arity; it was
corrected with explicit field names before all successful checks. No economic
access, input resampling or gate relaxation occurred.

No new public market requests, account access, credentials, orders, funds,
on-chain work, protected capture access or automation changes. The exact October
8 weather rejection was reused; a next date alone offered insufficient new
information to justify another similar capture. Historical financial artifacts
and both edge ledgers remain unchanged: 202 observations, 65 hypotheses,
37 scoped mechanisms, **zero qualified stable profitable strategies**.
Shared legacy Git identity violations remain previously surfaced and unrewritten;
this checkpoint uses `AI agent <>` for author and committer.
