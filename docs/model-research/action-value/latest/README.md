# Round 30: LightGBM hurdle ensemble abstained

**Rejected without trading authority.** All twelve threshold-selection stress simulations were positive after configured costs, and the best reused policy-validation short tails were positive, but every threshold candidate contained only 1 to 12 trades and failed the precommitted minimum-count gate. Signals meeting pre-threshold controls appeared only for Conservative (20), Regular (109), Aggressive (212); the 12 resulting threshold candidates all failed the stress-test acceptance criteria, so no reused policy-validation simulated trade, development prediction or profile evaluation, leverage, or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | 0.594 (long) |
| Best policy-validation stress ROC AUC (reused window) | 0.613 (short) |
| Best policy-validation top-100 mean net return (reused window) | +17.88 bps (short) |
| Largest pre-threshold eligible signal set | 212 / 28,581 (aggressive) |
| Thresholds evaluated / accepted | 12 / 0 |
| Policy-validation simulated trades (reused window) | 0 |
| Positive threshold-selection stress simulations | 12 / 12 |
| Maximum candidate trades / required minimum | 12 / 20 |
| Best threshold-selection stress net return | +135.76 bps from 5 trades |
| Authorized / live-executed trades | 0 / 0 |
| Numeric-contract replay | Identical boosters, forecasts, roles, and threshold-selection results |
| Development governance | Labels materialized; predictions and profile metrics not evaluated; window treated as consumed |


**Research-governance warning:** the policy-validation window has been reused across rounds and is selection-contaminated. It is not independent out-of-sample or terminal evidence.

**Development-window correction:** the shared target builder materialized barrier labels for the declared development dates. Predictions and profile metrics were not evaluated, but the window is conservatively treated as consumed. The reserved 2023-07-07 terminal day was outside the dataset and remains untouched.

![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Threshold-selection stress economics](charts/threshold-economics.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 229,001 valid event labels from 877,894 exact-BBO rows. The simulation uses 900 s positions, 100 ms paths, 750 ms total latency, and 12 bps configured taker round-trip cost. Round 30 revision 2 restores the sealed float-valued JSON types; its target hash matches Round 26, while every booster string and substantive metric exactly reproduces revision 1.

The hurdle ranking produced promising but statistically insufficient after-cost tails: threshold-selection stress ROC AUC reached 0.594, and the best reused policy-validation top-100 mean was +17.884 bps, but 5 of 8 displayed top-100/top-500 means were negative and no threshold was accepted. The architecture is a materially stronger research lead, but the next precommitted change must test broader chronological support and stability without lowering minimum trade counts, execution costs, drawdown limits, or abstention controls. Development labels were materialized, but development predictions and profile metrics were not evaluated. The reserved 2023-07-07 terminal day remains untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [replay integrity](replay-integrity.json) | [governance correction](governance-correction.json) | [integrity report](report.json)
