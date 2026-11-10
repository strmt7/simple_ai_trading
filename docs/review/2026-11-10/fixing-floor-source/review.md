# Partial-fixing source feasibility

A prepaid linear call can have a positive contractual payout floor before
expiry when part of its settlement average is irrevocably fixed. If qualified
nonnegative weights sum to one, the observed raw-price contribution is `L`,
and remaining prices are nonnegative, then settlement price is at least `L`.
The gross call floor is `q * max(L - K, 0)`. Executable acquisition cost, fees,
capital alternatives, currency, account treatment and exceptional settlement
still precede any profit claim. No venue weighting rule is supplied by this
algebra.

## What the retained sources establish

The exact native Crawl4AI extraction of the retained Clearing Procedures was
reused without another PDF download. On PDF page 37, Procedure 122 explicitly
specifies one-second index samples for delivery futures. On page 38, Procedure
128 defines an ordinary option's preceding 30-minute arithmetic mean without
specifying that sampling schedule. Procedure 131 separately specifies one-second
sampling for removal. Those other-product/event clauses do not qualify an
ordinary-option weight schedule. The document's effective version and native
account/API applicability remain unverified, as in the original
[source review](../pdf-source/review.md).

The source-bound code search found the supplied long-option floor and declared-
weight settlement tracker. The inspected Polymarket Round 25 adapter consumes
a named 30-second feed; Round 29 explicitly does not reconstruct the Chainlink
TWAP. These components alone do not qualify the raw observations and weights
of a terminal settlement window. This scoped lookup is not an exhaustive
absence claim or a rejection of their predictive features.

## A rolling average is not an accrued raw-price contribution

Two constructed positive paths have the same two observed three-point rolling
means, both 100. For raw indices `[-2,-1,0,1,2,3]`, the paths are:

| Path | Raw values | Mean at 0 | Mean at 1 | Terminal mean over 1–3 |
| --- | --- | --- | --- | --- |
| A | 100, 100, 100, 100, 1, 1 | 100 | 100 | 34 |
| B | 1, 298, 1, 1, 1, 1 | 100 | 100 | 1 |

A call with constructed strike 20 pays 14 on A and zero on B. More generally,
replace each 1 in B with positive `e` and 298 with `300 - 2e`: the observed
means stay 100 while the terminal mean approaches zero. Published rolling
averages alone therefore cannot supply the assumed positive accrued-prefix
floor. These exact [controls](controls.json) are algebra, not vendor samples,
observed prices, an economic test or independent validation.

## Bounded primary documentation attempt

One separately frozen navigation targeted the official options market-data
documentation, selected for index/estimated-exercise calculation semantics.
Ninety-eight selected source-plan/result/journal metadata files had no matching
stable path; raw/protected/tool environments were excluded before reading.
That lookup has limited scope and does not prove universal URL novelty.

The existing isolated provider passed read-only dependency/repair verification.
The source returned actual **HTTP 202**, native success true, **one Markdown
character**, and **4,512 retained bytes**. It fails technical admission; no schema,
sampling, fee or settlement fact is admitted from it or from search snippets.
The frozen failure and journal remain intact. No alias, market metadata, quote,
account, order, fit or benchmark followed.

Current decision: require applicable native weighting and raw-observation/clock
proof before evaluating a partial-fixing floor. Do not average rolling-average
publications, import futures/removal rules or invent a sample count. The source
gap is not a market rejection or a family-wide impossibility proof. Counts stay
205 observations / 65 hypotheses / 37 scoped mechanisms / 0 qualified stable
edges. This inquiry is not a new terminal economic observation.
