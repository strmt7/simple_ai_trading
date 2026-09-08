# CXMT post-change funding: exact study rejected

The September 4 trigger's first twelve Binance four-hour settlements were
collected once on September 8, with their 48 matching Polymarket hourly rows.
Both public GETs returned HTTP 200; raw responses and intent/completion journals
are retained beside the [frozen plan](plan.json) and [canonical result](result.json).
The receiving direction was fixed before access: long Polymarket, short Binance.

All twelve gross equal-notional funding-rate differences were negative.

| Frozen role | Settlements | Gross bp | After execution and capital bp | After additional quote stress bp |
| --- | ---: | ---: | ---: | ---: |
| Training | 6 | -4.3037 | -27.043426 | -37.043426 |
| Validation | 3 | -5.6561 | -27.025963 | -37.025963 |
| Test | 3 | -4.1478 | -25.517663 | -35.517663 |

Displayed values are rounded; JSON retains exact Decimal calculations. Each
role charges 20 bp execution and 500 annual bp per leg over its own elapsed
hours (two legs, 8,760-hour year). The extra 10 bp quote-unit stress, 75% positive
settlement requirement and full-cost drop-one check were also frozen before
access. No role passed. These are screening hurdles, not measured account fees.

This is an equal-notional funding-rate proxy, not realized fixed-base hedge PnL.
Actual mark paths, pUSD/USDT conversion, fees, margin, current instrument
eligibility and cross-regime stability remain unqualified. The first settlement
timestamp is after the rule change, but its four-hour accrual interval includes
15 pre-change minutes. Do not infer an entirely post-change accrual population.
No price, book, account, credential, order, fund or protected-partial access was
used. GPU training cannot repair the missing economics or create causal labels.

## Decision and verification

The exact sample is terminal: no refetch, orientation reversal, window extension,
Special-flow rescue or book/account escalation. This rejection does not prove
every funding strategy impossible. A new study requires a later material funding
cash-flow, fee, session, conversion or execution-architecture change, not merely
another date. Prior top-five and HK0625/SHEIN studies remain consumed.

Fifteen focused offline checks passed before capture. Final Ruff checks passed.
The registration tool reconstructed the economic result from retained raw bytes
and verified every frozen source binding, canonical contract/result hash and
raw receipt without network access. Each journal has exactly one intent and one
matching completion. No historical result or frozen implementation was changed.

Result SHA-256: `fb72b55226a32a45e568d9a9bf62a06e40bd7450e193f2ba52a93e3bdc8ee210`.
Registry: 37 mechanism scopes, 65 hypotheses, 193 terminal observations and zero
fully qualified stable profitable edges. Registry/audit self-hashes:
`97b48ded00c92bd356cd15700a1802e3cb4099af99fce1e930f55d04d6e349b7` /
`becf4054b7c8191e70e8bbcc4bcdc586bfe2929a5ba0e278c9380a5b96547087`.
