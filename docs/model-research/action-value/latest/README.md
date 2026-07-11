# Round 23: causal temporal-attention outcome model abstained

**Rejected without trading authority.** The bounded 30-second context improved the policy-window long top-100 mean net return under stress, but the gain did not persist in the calibration window or broader ranked tails, signal eligibility fell sharply, and every nonempty threshold-selection simulation lost money after stress costs. Signals meeting pre-threshold controls appeared only for Regular (2), Aggressive (17); Conservative produced none. The 8 resulting threshold candidates all failed the stress-test acceptance criteria, so no out-of-sample simulated trade, development access, leverage, or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | 0.627 (long) |
| Best out-of-sample stress ROC AUC | 0.611 (short) |
| Best out-of-sample top-100 mean net return | +0.94 bps (long) |
| Largest pre-threshold eligible signal set | 17 / 28,554 (aggressive) |
| Thresholds evaluated / accepted | 8 / 0 |
| Out-of-sample simulated trades | 0 |
| Authorized / live-executed trades | 0 / 0 |

![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 229,001 valid event labels from 877,894 exact-BBO rows. The simulation uses 900 s positions, 100 ms paths, 750 ms total latency, and 12 bps configured taker round-trip cost.

Probability-of-profit discrimination did not translate into an economically usable net-return ranking: threshold-selection stress ROC AUC reached 0.627, and the best out-of-sample top-100 mean was +0.940 bps, but 7 of 8 displayed top-100/top-500 means were negative and no threshold was accepted. The next precommitted change must test regime- or horizon-conditioned target formation and ranking stability without relaxing any risk control; the isolated positive policy tail is insufficient evidence of an edge. The development window and reserved 2023-07-07 terminal day remain untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
