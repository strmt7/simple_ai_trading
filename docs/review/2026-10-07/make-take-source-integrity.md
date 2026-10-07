# Maker input integrity and partial-cash counterexample

Reviewed October 7, 2026, against `a8411fb3f6c1e1cbc25b0617fe82f90a67d3c552`.
Exact source bindings and bounded verification are in the companion
[machine-readable review](make-take-source-integrity.json).

## Repair

The scenario builder checked hash length but did not invoke the existing fill
content validator. It now checks that contract before alignment. One shared
canonical entry payload preserves valid identities. The new entry validator checks
configuration, exact vector types/shapes, finite values, content hash, action/side
alignment, event chronology and passive/aggressive lifecycle invariants.

All three current entry consumers validate before accessing their downstream data:
targets before price paths, survival before features, and conditional payoff before
features. Conditional payoff also invokes the existing target validator. Valid
golden entry and target hashes remain unchanged. This protects supplied content;
it does not authenticate source origin or recover quantities the full-fill kernel
discarded. Existing historical source revisions/results remain in Git unchanged.

## Financial limitation reproduced

An entirely synthetic, explicit no-cancellation FIFO example has 104 printed units
at price 100, 100 ahead, and an intended own order of ten. The virtual helper gives
four partial units and six remaining; the legacy full-fill kernel gives no complete
fill and discards printed quantity. Its eligible censored target reports zero.
Closing the four units at 98 produces 400 acquisition cash and 392 proceeds;
configured entry cost plus slippage is three basis points (0.12), and exit cost
plus slippage six (0.2352). Net cash is **-8.3552 quote units**, not zero.

These are synthetic quantities, prices and base-scenario cost assumptions, not
native fees, owned fills, observed trades or performance. No historical target was
repriced. The old label cannot reconstruct partial inventory from discarded data.

## Focused evidence

- Before the fill/entry repair, 11 of 12 initial cases failed; the counterexample
  passed and exposed the existing label limitation. Console evidence only.
- Before the two panel guards, three targeted cases failed by touching downstream
  input before validation. Console evidence only.
- Final affected suite: 85 passing cases across source integrity, entries, targets,
  survival, payoff, replay and forward evaluation. Ruff checks/format checks passed
  for the five touched source files and two tests. No full-suite or hosted-CI claim.
- Tamper cases include recomputed hashes with invalid lifecycle; a matching hash
  alone is not an admission criterion. Golden identities are bound in the JSON.

## Next useful work and boundaries

Implement a native or qualified simulated partial-quantity cash ledger, protective
exits starting with the first partial fill, base-quantity matching and hedge/orphan
costs with explicit maker/taker role evidence. Do not fit these legacy proxies.
Feature-batch content identity remains open; independently qualified venue origin,
queue reconstruction, print uniqueness and actual fee coverage are also required.
The repair does not make datasets cash-qualified or trading institution-grade.

No market-data/financial-source requests, credentials, accounts, orders, protected partial data,
model fits, dependency changes, GPU/timing benchmarks or temporary campaigns were
used. No market retry trigger was consumed. Research totals remain 201 economic
observations, 65 hypotheses, 37 mechanism scopes and zero qualified stable edges.
GitHub contributor identity was refreshed read-only before publication. Shared
historical identity violations already surfaced remain unrewritten.
