# Round 29: 1800-second horizon outcome model abstained

**Rejected without trading authority.** The longer fixed horizon admitted substantially more signals under the regular and aggressive profiles, but calibration net-return ranking deteriorated, all eight threshold candidates lost after stress costs, and the least-negative trace was materially worse than the 900-second Round 26 baseline. Signals meeting pre-threshold controls appeared only for Regular (58), Aggressive (612); Conservative produced none. The 8 resulting threshold candidates all failed the stress-test acceptance criteria, so no reused policy-validation simulated trade, development access, leverage, or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | 0.589 (long) |
| Best policy-validation stress ROC AUC (reused window) | 0.602 (short) |
| Best policy-validation top-100 mean net return (reused window) | -2.02 bps (long) |
| Largest pre-threshold eligible signal set | 612 / 28,340 (aggressive) |
| Thresholds evaluated / accepted | 8 / 0 |
| Policy-validation simulated trades (reused window) | 0 |
| Authorized / live-executed trades | 0 / 0 |


**Research-governance warning:** the policy-validation window has been reused across rounds and is selection-contaminated. It is not independent out-of-sample or terminal evidence.

![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 227,011 valid event labels from 877,664 exact-BBO rows. The simulation uses 1800 s positions, 100 ms paths, 750 ms total latency, and 12 bps configured taker round-trip cost.

Probability-of-profit discrimination did not translate into an economically usable net-return ranking: threshold-selection stress ROC AUC reached 0.589, and every displayed top-100 and top-500 realized mean net return remained negative. The 1800-second fixed horizon is rejected under the retained taker-cost model; the next precommitted change must test state-conditioned horizon selection or genuinely multi-horizon targets without weakening fees, slippage, latency, stop barriers, or risk controls. The development window and reserved 2023-07-07 terminal day remain untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
