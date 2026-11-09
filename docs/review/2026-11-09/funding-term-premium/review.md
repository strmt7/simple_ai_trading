# Funding cash versus dated-future premium

This round investigates a capital/execution variant of existing carry research:
match opposite positions in a dated future and a perpetual instead of funding
a spot purchase. It is a conditional financial hypothesis, not a new accepted
edge, a proven margin saving or permission to reopen a consumed study. The
existing mainnet account/fee authority and all exact retry triggers remain intact.

## The cash question

Assume two linear contracts with the same base quantity, quote/payment unit and
holding interval. Let `D` be the dated-future price, `P` the perpetual price,
`q > 0` the matched base quantity, and `b = D - P`. For long dated/short perpetual,
the price component is exactly:

```text
q * [(D_T - D_0) - (P_T - P_0)] = q * (b_T - b_0)
```

Adding actual signed **additional non-price** cash flows from both legs, and
every separately identified cost, gives:

```text
net surplus = q * (b_T - b_0)
              + additional_cash_dated + additional_cash_short_perpetual
              - execution - financing - opportunity - other_costs
```

These are algebraic identities under the stated contract assumptions, not
facts extracted from a blocked paper or current market PnL. Do not assume the
dated contract has zero additional cash obligations until its terms qualify.
Variation-margin or final-settlement receipts already represented by the price
component must not be added again as additional cash. Reconcile actual receipts
against each component rather than counting both marked profit and its payment.
If it settles at an index `I_T`, the terminal spread is `I_T - P_T`; a perpetual
has no matching expiry merely because the dated future does. Never force that
exit spread to zero. A cash-settled dated leg also needs its actual settlement
quantity, price, fee and clock, rather than an invented executable close.

For this orientation, positive funding alone is insufficient: the cash must
exceed `q * (b_0 - b_T)` plus all costs. For the opposite orientation, reverse
both positions and their actual cash consequences, not just a reported return.
Any orientation choice belongs to a prospectively fixed training role, never
whichever sign wins in validation or test.

The economic target is accumulated **cash funding minus paid term premium and
exit basis**, with transition/unwind costs. A funding-rate point forecast or
annualized headline does not represent it. A useful future model needs the joint
distribution of cumulative payments, terminal basis, partial-fill exposure and
required collateral, including discrete payment/settlement times.
Under the retained fixed-base linear funding law, short-side funding cash scales
each payment by its own settlement mark. Forecasting a rate mean alone misses
the mark/rate dependence and any changes in actual inventory at payment time.

The potentially useful difference is capital and execution: a funded spot
purchase is replaced by another margined contract. That does not prove lower
usable capital. Same-venue positions do not prove portfolio offsets, shared
liquidation treatment or permission to use one leg's profits as collateral for
the other. Measure the feasible collateral path and capital return on an equal
initial-wealth alternative. Opportunity cost applies to reserved capital over
time, not an assumed leverage multiple; reconcile it against financing/cash
receipts to prevent double charging. Preserve gap, separate-liquidation,
settlement, outage and emergency-unwind risk.

## Retained source feasibility

The [source receipt](source-result.json) binds a zero-network inventory using
the unchanged vendor catalog and two native depth responses. The catalog lists
1,011 perpetuals and 53 futures. Fifty BTCUSDT/ETHUSDT dated candidates appear
across its complete history; exactly four declared symbol spans contain the
fixed October 1, 2026 research day:

| Candidate identifier | Declared coverage begins |
| --- | --- |
| BTCUSDT_261225 | June 26, 2026 |
| BTCUSDT_270326 | September 25, 2026 |
| ETHUSDT_261225 | June 26, 2026 |
| ETHUSDT_270326 | September 25, 2026 |

All four list book-ticker/depth dataset types. This is vendor metadata only:
actual files, exact contract multipliers/payment units, settlement identity,
native costs, account margin compatibility and executable capacity remain
unqualified. No dated-contract quote was requested or inspected. There is no
SOL dated counterpart claim. The identifiers are evidence, not runtime campaign
or hardcoded trading selections.

The offline timing inventory also quantifies the six previously retained books:

| Book | Spot snapshot delay after first increment | Perpetual delay | Earlier increments, spot/perpetual |
| --- | ---: | ---: | ---: |
| BTCUSDT | 1.5758678 s | 0.7149232 s | 16 / 28 |
| ETHUSDT | 0.5852877 s | 0.4874596 s | 7 / 19 |
| SOLUSDT | 2.3174256 s | 0.5075618 s | 23 / 18 |

These are collector-local receipt differences, parsed as exact integer
nanoseconds. They are neither measured exchange latency nor evidence of
inter-collector clock synchronization. A historical adapter must preserve source
ordinal order, snapshot availability and unknown clock offsets. Do not pass a
zero offset into the local snapshot gate as if it had been measured for this
third-party source. No common executable cross-book snapshot is qualified.

## Research access and decision

One frozen two-query search discovered primary research URLs. Older arXiv papers
were already cited in existing designs and were not fetched again. A distinct
EPFL institutional record was selected under a new two-page contract, but
Crawl4AI returned HTTP 429, native failure and 464 extracted characters identifying
anti-bot protection. Source admission failed. The SSRN second page was forbidden
and not requested. No aliases, alternative reader or challenge bypass followed.
No paper theorem, empirical return or publication-date claim is admitted from
the search snippets or failed extraction.

Pre-publication digest verification caught precision loss in the derived summary's
nanosecond integers during a JavaScript numeric round trip. Native captures and
the private Python result were unchanged. The faulty export was retained
privately, and the summary was regenerated from exact private integers while
forwarding serialized JSON text. The provenance policy now requires that lossless
boundary; the correction is not a source retry or a rewrite of an old result.

The search/offline intents record the last observed UTC instants before their
creation; the native completion journals and tool ordering retain actual later
execution times. No outcome-sensitive access preceded either intent. Preserve
the originals and use fresh observed time on any later contract.

Next, compare this variant's required evidence with the existing carry boundaries
before any new data access. It needs a genuinely eligible prospective paired
population, qualified linear contract units and native settlement/cost semantics,
causal clocks, fixed role/orientation/capital rules and untouched chronological
evaluation. Previously observed funding panels may only support explicitly
exploratory interpretation under their existing restrictions. Do not reprice them
into independent validation or bypass the existing account gate with this variant.

This source-only round adds **zero economic observations**. Counts remain
203 observations / 65 hypotheses / 37 scoped mechanisms / 0 qualified stable
edges. No new model fit, test suite, price/size inspection, native market request,
credential, account, order or funds access occurred. Raw responses, journals and
the failed controller remain ignored and immutable; the controller is inert.
No new persistence or GitHub push was needed. The whole goal remains active.
