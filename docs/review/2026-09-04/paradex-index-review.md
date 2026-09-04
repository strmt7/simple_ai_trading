# Paradex/Binance funding: complete index study, insufficient cost headroom

The new, non-overlapping fixed-base study completed. All nine chronological
role subtotals have positive gross par-valued funding differences, but none
covers the frozen 20-bp execution allowance alone. Even charging that allowance
only **once across the entire 8⅔-day window** leaves every asset negative.
No book, basis, account, order or protected-capture request follows.

| Asset | Training gross bp | Validation gross bp | Test gross bp | Whole-window gross bp | Whole window less one 20-bp execution allowance |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.3494 | 0.7858 | 1.0097 | 3.1169 | -16.8831 |
| ETH | 7.3736 | 2.4484 | 1.5068 | 11.2099 | -8.7901 |
| SOL | 7.5875 | 2.1798 | 1.8630 | 11.6076 | -8.3924 |

Gross is a **USDC/USDT par-valuation diagnostic**, not observed conversion or
account PnL. Each role uses its preceding Binance funding mark as a reference
denominator; the whole window uses its own first reference, so displayed role
basis points should not simply be added. The whole-window comparison includes
training data and is rejection-only, not another validation result. The 20 bp
is a preregistered execution allowance, not a measured commission. Actual
fees, basis, liquidation, capital, custody and conversion remain unqualified.

## Why this was a distinct useful test

The previous rank-59 study ended at incomplete paginated sources, without
evaluating funding. Its source window ended August 26 at 17:00 UTC. This
study uses August 27 00:00 through September 4 16:00 nominal boundaries and
all three preselected BTC/ETH/SOL assets. No failed old response was opened,
repaired, paginated or used as a new holdout. The exact 84 request URLs had no
retained documentation/contract/journal match before access.

Two separately frozen public documentation GETs established the index route:

- The official [funding mechanism](https://docs.paradex.trade/risk/funding-mechanism)
  describes a cumulative settlement-asset index per base unit and realizes a
  fixed position's funding from its signed index change. Its separate unrealized
  expression includes an oracle valuation factor; do not multiply the realized
  settlement-unit cash by that factor again.
- The official [public history API](https://docs.paradex.trade/api/prod/markets/get-funding-data)
  exposes `funding_index`, `created_at`, market and funding period, bounded
  millisecond queries, page-size limits and continuation indicators. It does
  not promise history cadence or row order. Thus this study admitted only
  complete bounded windows and selected the earliest timestamp, not first row.

These current sources do not establish their own historical effective version,
actual account credits or an absence of historical corrections. They support
the public prefilter's interpretation, not promotion. Original HTTP bodies
were 1,007,567 bytes for the mechanism and 1,207,581 for the API; access occurred
at 23:00:11–23:00:16 UTC on September 4. Their exact hashes remain in the
unchanged study contract and durable request receipts.

GitHub rejected the initial unpublished checkpoint because the full-site HTML
embedded AWS access-key patterns outside the relevant page text. The initial
PEM/assigned-credential pattern check was insufficient: a provider-format check
found 44 occurrences in each body. Neither their validity nor ownership was
tested, and push protection was not bypassed. Both original HTML bodies were
removed from the working tree and publishable commit. The mechanically selected
secret-free main sections (11,580 and 1,465 bytes), extractor identity, original
response hashes and explicit disposition are retained in
`paradex-index-source/`. The original full bodies **cannot** be reconstructed
from those sections; the verifier reports this exception rather than claiming
all original documentation bytes remain available. All 84 economic responses,
frozen contracts, funding calculations and historical results are unchanged.
No documentation alias or repair request was made.

## Frozen measurement and result

[Contract](paradex-index-contract.json):
`0bedfd6efc1fca0f7a8657d290e51f58b4c435eda89f2c54ee680b11902d166b`.
[Result](paradex-index-study/result.json):
`f2aa72d9d65d28bdd0f012deb85d2e65c5085d7c62ab9c69aee916401b6f99f9`.

- 27 index boundaries and one Binance funding history per asset: **84 GETs,
  1,308,306 raw bytes, 78 intervals**, no redirects, retries or pagination.
- Each index window begins 1,001 ms after the nominal boundary and ends at
  301,000 ms. Selected actual offsets were 1,684–5,654 ms. Every Binance
  settlement lies strictly inside its actual index interval, rather than being
  assumed simultaneous with it. No index interpolation or rate extrapolation.
- Funding cash uses one fixed base unit: Paradex short index delta in USDC
  versus Binance long funding rate times its actual settlement mark in USDT.
  These are research reference quantities, not filled order sizes.
- Chronological roles contain 13/6/7 intervals. Training alone selected short
  Paradex / long Binance for each asset. Later roles were not reoriented.
- All original cost stresses were retained: 20 bp execution, 25 quote-unit,
  25 custody/latency, 10 extra reserve, and 10% annual total-two-leg capital
  over actual elapsed time. The old 10-bp sampling stress is now explicitly an
  extra reserve; direct index differences did not justify dropping it after
  seeing results. Every asset failed every role's net gate. BTC additionally
  failed training positive-interval frequency and validation half stability.

The source study ran 23:09:52–23:11:12 UTC. Its process exited normally.
The separate offline verifier reconstructs all 84 market-response hashes, request journals,
time selections, funding cash flows, orientations, roles and the cost-only
comparison, and separately checks the documentation dispositions. No GPU
workload, historical database scan or model fit was needed.

## Interpretation and next action

There is a small observed gross spread in this sample, not a demonstrated
economically tradable edge. The execution-only comparison avoids attributing
failure solely to the larger capital/custody stresses or repeated role-level
entry costs. It is not proof that all future funding spreads have negative
expected value, nor permission to assume zero fees and call the sample profitable.

The exact population is consumed, including every frozen post-boundary window.
Do not refresh, roll it forward mechanically, reorient, drop an asset, or
request books to rescue it. The existing literal non-overlap/material-change
trigger remains, with an explicit information-gain case required for another
study. A future survivor still needs instrument identity, genuinely executable
costs and basis, currency conversion, margin/liquidation and independent
prospective persistence. The same cash-unit discipline applies to Binance/
Polymarket Perps comparisons; their separate frozen windows remain untouched.

Canonical counts are now **37 accepted scopes, 65 hypotheses, 188 terminal
observations, zero stable current account-qualified after-all-cost edges**.
The registry and durability audit were updated together; their exact previous
bytes are retained under `paradex-index-publication/`. No historical result
or accepted scope was modified.
