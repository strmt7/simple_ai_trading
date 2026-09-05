# Organic conversion fee headroom: narrow, unqualified

This is an explicitly post-capture, exploratory reuse of the twelve retained
September 5 triangle-window responses. It is not a second validation population
or a repair of that failed cycle screen. No new network/account/fee/depth request,
order, training run or protected-data access occurred. Prior results and all
retry triggers remain unchanged.

The [plan](plan.json) binds every raw input and implementation. The
[complete result](result.json) contains all **288 comparisons**: 24 distinct
one-intermediary paths across all twelve samples, covering every directed
source/target pair among BTC, ETH, SOL and USDT. No selected-only output.

For an independently required source-to-target conversion, let `d` be direct
output per source unit and `a*b` indirect output. Gross relative output is
`R=a*b/d`. Under proportional received-asset fees, net relative output is
`R*(1-f1)*(1-f2)/(1-fDirect)`. When the same fee `f` applies to all three
symbols, indirect beats direct only if `f < 1-1/R`. Direct execution pays one
fee and indirect pays two; subtracting two fees from gross advantage without
accounting for the direct fee would compare different economic baselines.

These formulas do not model BNB-paid fees, minimum/flat fees, rounding, depth,
partial completion or sequential risk. The fee levels below are hypothetical,
not this account's commissions. All ratios are in target units, not realized
USD PnL, and presume the compared quotes are simultaneously feasible.

| Uniform hypothetical fee per leg, bips | Positive row comparisons | Routes positive in all 12 samples |
| --- | ---: | ---: |
| 0 | 113 | 3 |
| 1.2 | 52 | 0 |
| 2.4 | 17 | 0 |
| 10 | 0 | 0 |

No route remained above an additional 3-bip operational stress in all twelve
samples, even at zero fees. That stress is an assumption, not a measured loss.
The largest single gross comparison was 3.6967 bips for `ETH -> USDT -> SOL`
versus direct `ETH -> SOL` at sample 9. It is not a persistent opportunity.
The strongest minimum equal-fee ceiling was **0.6786 bips** for
`ETH -> USDT -> BTC` versus direct `ETH -> BTC`; its ceiling is conditional on
ignoring every other friction and the retained quote-timing limitations.

## Resource decision

Keep rank 44 as an overlay for genuinely independently required organic flow,
not stand-alone arbitrage or a static route to trade. This diagnostic does not
satisfy its account/organic-flow trigger or authorize another capture. No route
is promoted. The narrow headroom makes fee qualification and executable
extra-leg completion essential; a larger model or repeated REST sampling would
not supply those missing facts. Counts remain 37 accepted scoped mechanisms,
65 hypotheses, 190 terminal observations and zero fully qualified stable edges.

Fourteen focused tests cover symmetric/asymmetric fee accounting, invalid
rates/fees, exhaustive route enumeration and complete retained reconstruction.
The pure helper can evaluate supplied asymmetric fee assumptions, but this
frontier uses only its declared uniform scenarios. It is not an account adapter.
