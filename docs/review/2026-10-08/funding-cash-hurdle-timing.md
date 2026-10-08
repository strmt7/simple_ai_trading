# Funding cash in hurdle labels and second-flow timing

## Implemented scope

Baseline: `5292c8b5566e134e6b1f05d04226a935bcf091d7`. The previous goal turn
implemented two concrete builder routes and preserved their historical defaults;
it was progress toward trustworthy profitability measurement, not proof of alpha.

The current hurdle and second-flow builders now accept the same explicit
`FundingCashLabelSeries` inputs used by stop-time/barrier. Four of five audited
routes therefore support the supplied marked-event law. No new native dataset,
fit, profitable-edge claim or historical-result rewrite follows.

Hurdle target classes use separate side-specific adverse cash values. Arrays
retain lower/upper cash and uncertain-event counts in short/long order for each
horizon. The compatibility funding field retains its long-debit convention;
the paired replay selects the actual side's cash bound, never a sign flip of
that compatibility field. Incomplete/conflicting metadata rejects. Serialized
`DerivativesSourceEvidence` includes cash provenance only for the new cash path;
legacy serialization stays unchanged. Features and chronological role masks
are unchanged. Real historical marks remain unavailable for the failed old batch.

Second-flow timing calculates mark cash in six symbol/side batches, not a new
rational event calculation for every delay option. Legacy calls do not allocate
the extra cash scratch arrays or change their labels. Explicit cash bypasses the
rate-sum helper, retains pre-storage float64 utility and stores adverse net/cash
and upper cash outward in float32. Cash metadata keeps the supplied certificate,
mark/rate/source digests and declared coverage. The validator rejects missing
provenance/bounds, invalid shape, nonfinite/ordered bounds, invalid counts or a
conflicting debit sign. Funding marks affect labels only, not decision features.

These are conditional fixed-base, entry-normalized label bounds, not certified
fills or native account cash. Exact exchange origin, complete population/clock
coverage, quantity semantics and after-all-cost execution require independent
qualification before training/admission. The old frozen Round 38/42 role windows
are preserved; their future data/target schemas require distinct evidence, not
reuse of a later successful mark page to patch an older missing population.

## Stateful sizing limitation

The unchanged `build_stateful_hourly_dataset` computes each hour's price/funding
target relative to that hour's entry price. `replay_stateful_policy` and
`replay_always_long` sum position times that target while charging only changes
in a -1/0/1 position indicator. This is not an actual fixed-base inventory ledger.

A deterministic synthetic control uses prices 100 -> 200 -> 100 and one base
unit. Fixed-unit gross cash is exactly zero. Hourly entry-relative returns are
+10,000 and -5,000 bps; the existing always-long replay reports +4,988 bps after
its illustrative two 6-bp transition costs. The difference is not a market edge.
Maintaining the initial 100-quote notional instead requires selling 0.5 base
units at the interior 200 price; its fees, spread and latency are absent from
that position-indicator transition count. This does not reject a properly
specified constant-notional strategy; it identifies the missing sizing/cost law.

The current signed forecast target also cannot represent both asymmetric
adverse funding outcomes by a sign flip. Do not merely substitute mark-weighted
rates into it. Next retain actual base quantities, entry notionals and separate
side cash, and account for every quantity change before this replay or training
can claim cash qualification. Old source/results are not rewritten.

## Verification and integrity

- One affected-domain run passed 155 checks, including 20 new builder/replay/
  validator cases. An additional isolated stateful sizing control passed, giving
  **156 distinct checks and 21 new cases**. Ruff passed on five touched Python
  files. There was no full-suite rerun, fit, model inference or GPU benchmark.
- Signed rates, both sides, boundary entitlement, both public builder routes,
  direct no-proxy controls, future-mark feature invariance and metadata failures
  are covered. The sizing control is synthetic arithmetic using the current
  replay, not an observed return, backtest population or strategy comparison.
- Source digests and exact verification commands are in the paired
  [implementation receipt](funding-cash-hurdle-timing-implementation.json).
- Financial registry/audit counts and artifacts stay unchanged: 202 observations,
  65 hypotheses, 37 mechanism scopes, zero qualified stable profitable edges.
- No market/source capture, protected partial data, credential, account, order,
  funded action, automation or unrelated host process was touched. A fresh
  anonymous GitHub contributor audit is read-only publication administration;
  known shared-history AI placeholder identities remain surfaced violations,
  not permission for a force rewrite. New identities use `AI agent <>`.
