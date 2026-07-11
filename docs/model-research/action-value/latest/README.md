# Round 19: depth-normalized order-flow model abstained

**Rejected without trading authority.** Depth-normalized aggressive order-flow inputs increased signal counts under the aggressive risk profile, but every threshold-selection simulation lost money after stress costs. Signals meeting pre-threshold controls appeared only for Aggressive (55); Conservative and Regular produced none. The 4 resulting threshold candidates all failed the stress-test acceptance criteria, so no out-of-sample simulated trade, development access, leverage, or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | 0.626 (long) |
| Best out-of-sample stress ROC AUC | 0.608 (short) |
| Least-negative out-of-sample top-100 mean net return | -5.99 bps (long) |
| Largest pre-threshold eligible signal set | 55 / 28,581 (aggressive) |
| Thresholds evaluated / accepted | 4 / 0 |
| Out-of-sample simulated trades | 0 |
| Authorized / live-executed trades | 0 / 0 |

![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 229,001 valid event labels from 877,894 exact-BBO rows. The simulation uses 900 s positions, 100 ms paths, 750 ms total latency, and 12 bps configured taker round-trip cost.

Probability-of-profit discrimination did not translate into an economically usable net-return ranking: threshold-selection stress ROC AUC reached 0.626, while every top-100 and top-500 realized mean net return remained negative. The next precommitted model change must address direction-specific net-return ranking and probability calibration rather than add further depth-normalized inputs or relax the risk controls. The development window and reserved 2023-07-07 terminal day remain untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
