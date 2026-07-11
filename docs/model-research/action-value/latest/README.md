# Round 28: sampled aggregate-depth outcome model abstained

**Rejected without trading authority.** The added sampled 1% and 5% depth shape improved several calibration and broader ranked-tail diagnostics, but the best out-of-sample long top-100 mean deteriorated, all eight threshold candidates lost after stress costs, and the least-negative aggressive trace was materially worse than the depth-free Round 26 baseline. Signals meeting pre-threshold controls appeared only for Regular (1), Aggressive (42); Conservative produced none. The 8 resulting threshold candidates all failed the stress-test acceptance criteria, so no out-of-sample simulated trade, development access, leverage, or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | 0.605 (long) |
| Best out-of-sample stress ROC AUC | 0.603 (short) |
| Best out-of-sample top-100 mean net return | -5.73 bps (long) |
| Largest pre-threshold eligible signal set | 42 / 28,554 (aggressive) |
| Thresholds evaluated / accepted | 8 / 0 |
| Out-of-sample simulated trades | 0 |
| Authorized / live-executed trades | 0 / 0 |
| Current sampled aggregate-depth features | 867,009 / 877,894 (98.76%) |


![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 229,001 valid event labels from 877,894 source-bound BBO/trade rows with sampled aggregate-depth features. The simulation uses 900 s positions, 100 ms paths, 750 ms total latency, and 12 bps configured taker round-trip cost. Official sampled 1% and 5% cumulative notional bands were current for 867,009 rows; 10,885 rows were explicitly masked after 60 seconds. These data are not a full event-level order book and provide no queue-position or maker-fill evidence.

Probability-of-profit discrimination did not translate into an economically usable net-return ranking: threshold-selection stress ROC AUC reached 0.605, and every displayed top-100 and top-500 realized mean net return remained negative. Static sampled aggregate depth is rejected as a sufficient edge; the next precommitted change must target cost-aware action formation or higher-frequency depth dynamics, and maker-order economics remain blocked until event-level queue evidence can support fill modeling. The development window and reserved 2023-07-07 terminal day remain untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [depth coverage](depth-coverage.json) | [integrity report](report.json)
