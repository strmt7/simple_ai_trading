# Round 21: pairwise net-return ranking model abstained

**Rejected without trading authority.** Sampled pairwise net-return ranking improved several discrimination and short-side tail diagnostics, but worsened the best out-of-sample tail and eliminated threshold-selection eligibility across all risk profiles. All three risk profiles had zero signals meeting pre-threshold controls, so no threshold, out-of-sample simulated trade, development access, leverage, or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | 0.629 (long) |
| Best out-of-sample stress ROC AUC | 0.609 (short) |
| Least-negative out-of-sample top-100 mean net return | -15.29 bps (long) |
| Largest pre-threshold eligible signal set | 0 / 28,581 (none) |
| Thresholds evaluated / accepted | 0 / 0 |
| Out-of-sample simulated trades | 0 |
| Authorized / live-executed trades | 0 / 0 |

![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 229,001 valid event labels from 877,894 exact-BBO rows. The simulation uses 900 s positions, 100 ms paths, 750 ms total latency, and 12 bps configured taker round-trip cost.

Probability-of-profit discrimination did not translate into an economically usable net-return ranking: threshold-selection stress ROC AUC reached 0.629, while every top-100 and top-500 realized mean net return remained negative. The next precommitted change must restore calibrated positive expected-return separation while retaining net-return ordering and the existing risk controls. The development window and reserved 2023-07-07 terminal day remain untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
