# Round 27: 300-second horizon outcome model abstained

**Rejected without trading authority.** The shorter lifecycle improved probability-of-profit discrimination and calibration metrics, but positive outcomes became rarer, every displayed ranked tail remained negative, and all eight threshold candidates failed after stress costs; the least-negative trace contained one losing trade. Signals meeting pre-threshold controls appeared only for Regular (8), Aggressive (40); Conservative produced none. The 8 resulting threshold candidates all failed the stress-test acceptance criteria, so no out-of-sample simulated trade, development access, leverage, or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | 0.700 (long) |
| Best out-of-sample stress ROC AUC | 0.682 (short) |
| Best out-of-sample top-100 mean net return | -15.51 bps (short) |
| Largest pre-threshold eligible signal set | 40 / 28,737 (aggressive) |
| Thresholds evaluated / accepted | 8 / 0 |
| Out-of-sample simulated trades | 0 |
| Authorized / live-executed trades | 0 / 0 |

![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 230,393 valid event labels from 878,025 exact-BBO rows. The simulation uses 300 s positions, 100 ms paths, 750 ms total latency, and 12 bps configured taker round-trip cost.

Probability-of-profit discrimination did not translate into an economically usable net-return ranking: threshold-selection stress ROC AUC reached 0.700, and every displayed top-100 and top-500 realized mean net return remained negative. The 300-second fixed horizon is rejected under the retained taker-cost model; the next precommitted change must improve executable action design or cost-aware target formation without weakening fees, slippage, latency, or risk controls. The development window and reserved 2023-07-07 terminal day remain untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
