# Action-Value Round 9 Evidence

Status: **rejected**. This is checksummed Binance USD-M discovery evidence, not
a profitability, execution, or trading-authority claim.

- UTC window: 2023-08-14 through 2023-08-20 (now consumed for selection)
- Precommitted candidates: 12
- Statistical fit failures: 7
- Trained candidates: 5
- Unrejected candidates: 0
- Policy and selection trades: 0
- Design SHA-256: `a6ac6be9d4322f1b78a5894c72e131b5ef596712dfd2decaff32c969373e76e6`
- Corpus certificate SHA-256: `4d03bd2ae6e2b19f2fbdfb5bd6d3c0b3dc89020346cdb3ac435acc253c492edd`
- Implementation commit: `8a0eec2f56b8a4a727a5dacdea098ed51b9ba917`

The 60/120-second candidates and conservative 300-second candidate lacked the
minimum profitable/non-profitable class support after actual spread, 5 bps
taker fee per side, and 1 bps additional slippage per side. The five remaining
models produced some short-side positive predicted-edge rows, but every
non-overlapping threshold policy using them had non-positive realized
drawdown-adjusted utility on the policy segment, so abstention was financially
correct under the fitted policy. A post-round
diagnostic found a bounded-Newton calibration collapse; it is fixed separately
and does not retroactively alter this evidence.

## Charts

![After-cost performance](charts/after-cost-performance.svg)

![Forecast quality](charts/forecast-quality.svg)

![Action funnel](charts/action-funnel.svg)

![Research progress](charts/research-progress.svg)

The source tables are [candidates.csv](candidates.csv) and
[progress.csv](progress.csv). Every trained artifact SHA-256 and every fit error
is retained in `candidates.csv`; no zero-trade equity curve is fabricated.
Regenerate with `python tools/publish_action_value_discovery.py`.
