# Forward funding cash-label integration

## Scope and financial consequence

The previous substantive turn preserved a one-use NFL screen's resource failure;
it supplied a future design correction, not a price rejection or profitable edge.
The intervening wrap-up committed/pushed that checkpoint and paused at the user's
request. The October 8 explicit `Continue` resumes the active goal. No scheduled
task was created or edited in this continuation.

This checkpoint implements two of the five audited cash-label consumer routes:
`build_stop_time_payoff_dataset` and `build_barrier_payoff_dataset`. Optional,
explicit `funding_cash` inputs use `FundingCashLabelSeries`, which calls the existing
`funding_cash_bounds_for_events` exact fixed-base payment law. The former default
rate-sum calculations are preserved for historical reproducibility. Those
defaults are not cash-qualified labels and must not be used for new admission.

The retained October 7 cash-unit audit's synthetic 100 entry / 102 mark / 0.005
rate control is reproduced as 51 bps, not 50 bps, with the opposite sign for a
short. This is an accounting control, not a venue return or new market evidence.

## Binding, timing and numeric contracts

- Marked settlements must exactly match the caller's independent ordered
  time/rate population; missing marks, conflicting rates, invalid symbols,
  mutable populations and absent certificate digests reject. The consumer's
  existing rate state must also match inside the declared UTC coverage.
- A certificate digest is an evidence identifier, not a proof of authenticity
  or completeness. Native origin, actual certified coverage and the price/fill
  sources require independent qualification before a fit or financial claim.
- Funding is normalized by fixed-base entry notional, not constant-notional
  exposure. Entry and exit equality remain UNKNOWN; strictly interior events
  before the earliest supplied exit are modeled HELD. A triggered stop/take's
  exit spans its entire minute; its end is not a known fill timestamp. These
  bounds are conditional on the supplied holding-clock model, not bounds on
  exchange settlement delays, slippage, complete P&L or real account eligibility.
- Each side uses its own adverse cash bound. Unknown positive funding is not
  credited, while a possible debit is retained. Zero funding is not fabricated
  from missing rows. A genuinely empty population still requires independent
  certification; the numerical empty-case control proves no coverage claim.
- Exact rational event prefixes are converted outward into float64 intervals;
  array lookups avoid one rational computation per training row. Existing
  float32 storage rounds adverse cash/net labels downward and upper cash upward,
  rejecting nonfinite/overflowed storage. This controls funding rounding, not
  uncertainty in input trade prices or total execution accounting.
- Dataset identity additionally binds cash provenance, upper cash streams and
  uncertain-event counts. Legacy identity construction is unchanged. The
  barrier funding field keeps its historical debit/sign convention; multiply it
  by minus the selected side to obtain adverse received cash. Its last-axis
  order is short/long. Stop-time upper/count last-axis order is long/short.

## Verification and remaining work

The focused affected suite passed **128 checks**, including **55 new cases**:
signed rates and both sides; exact-law/vectorized parity; outward enclosure and
subnormal/overflow storage; invalid/missing/conflicting provenance and clocks;
timeout, intrabar stop, wrong-symbol rejection; both public builder routes and
legacy identity preservation. Ruff passed for all four touched Python files.
Direct monkeypatch controls also prove that explicit cash simulations never call
the legacy rate-sum helper; malformed complex/string rate schemas reject.
The full suite, models, performance timings and GPU benchmarks were not run.

The hurdle, second-flow and stateful builders remain rate proxies. In particular,
the stateful antisymmetric signed-target representation cannot be reused for
asymmetric adverse cash without retaining its two-sided cash consequences. Next
complete those routes, independently qualify native data/certificate coverage,
then replay a complete hedge after quantity-sensitive costs before the smallest
chronological fit. Existing partial-fill, financing, basis, margin, fee and
selection-risk gates remain controlling. No stable-profit or enterprise-complete
claim is made; registry observations/hypotheses/mechanism scopes/stable edges
remain **202 / 65 / 37 / 0**.

No new market/public-source requests, protected partial reads, credentials,
account mutations, orders, model fits, old-result rewrites or unrelated host
process changes occurred. The only external read was the required fresh anonymous
GitHub contributor audit for publication identity. Known historical AI placeholder
emails remain shared-history violations; they are not corrected by rewriting
shared history. New commits use literal `AI agent <>` for both identities.

Prior source bytes remain recoverable at checkpoint
`44030170df8e1bd8051fb4484b964d9a45e6bad3`; archived research artifacts are not
rewritten or requalified by these forward builder changes.

Startup routing is now explicitly limited to the first 80 lines of Agent Start;
older appended history is read only through needed links. An initial full dump
in this turn exposed the avoidable token cost that this instruction corrects.
The [implementation receipt](funding-cash-labels-implementation.json) binds the
changed Python bytes and focused verification scope. It is not market evidence.
