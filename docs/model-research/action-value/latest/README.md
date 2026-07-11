# Round 15: daily refits abstained

**Rejected safely.** All **84** prior-only threshold traces from **21** causal daily model fits failed the after-cost risk gates. No threshold was allowed to trade an evaluation day.

| Evidence | Result |
| --- | ---: |
| Least-negative policy calibration trace | -14.96 bps over 20 trades |
| Its maximum drawdown | 112.99 bps |
| Accepted thresholds | 0 / 84 |
| Evaluation trades | 0 |
| Research candidates | 0 |

![Prior-only calibration economics](charts/after-cost-performance.svg)

![Daily forecast quality](charts/forecast-quality.svg)

![Action funnel](charts/action-funnel.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 230,999 causal events from 878,025 exact-BBO rows. Traces use 750 ms latency and 12 bps configured taker round-trip cost. The development window is consumed; 2023-07-07 remains untouched.

No ROI or equity curve is shown because no evaluation trade occurred. Fixed-horizon traces still lack intrahorizon stop-loss paths, so this result cannot authorize trading, leverage, or a profitability claim.

Data: [candidates.csv](candidates.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
