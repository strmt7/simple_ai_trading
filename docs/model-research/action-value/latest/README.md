# Round 53: Executable Conditional Sign-Magnitude

> **Rejected consumed-development screen.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 53 factorized each side's executable payoff into six magnitude states and a magnitude-conditioned sign model. Eighteen new OpenCL LightGBM models used verified Binance USD-M BTC, ETH, and SOL event data; nine direct controls and sealed causal FinCast features were reused.

The CSM improved average expected-payoff rank to `0.051582` and every joint proper-score comparison, but mean calibration remained wrong. Its frozen 0.10% policy had 1 calibration trade at `-8.879308` stressed bps/trade. The 2 later trades averaged `+16.653357` bps, but calibration did not authorize them.

A separate fixed 54-rule rank-tail diagnostic removed the positive-EV requirement. Zero rules passed calibration. The least-bad calibration rule was `worst_seed` at `0.1%` coverage: 11 trades at `-1.673628` stressed bps/trade; its consumed result was `-4.022015`. Global correlation therefore did not establish executable top-tail alpha.

FinCast worsened joint log loss by `0.001392` and expected-payoff rank by `0.002586` versus the matched CSM control.

## Evidence

| View | Graph | Source |
|---|---|---|
| Executable support | [SVG](charts/executable-support.svg) | [CSV](support.csv) |
| Forecast quality | [SVG](charts/forecast-quality.svg) | [CSV](forecast.csv) |
| Frozen policy | [SVG](charts/policy-economics.svg) | [CSV](policy-grid.csv) |
| Rank-tail falsification | [SVG](charts/rank-tail.svg) | [CSV](rank-tail.csv) |
| Fixed-policy daily path | [SVG](charts/daily-equity.svg) | [CSV](daily-policy.csv) |
| FinCast uplift | [SVG](charts/ai-uplift.svg) | [CSV](ai-uplift.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`screen.json` and `rank-tail-screen.json` preserve the complete sources. `report.json` binds every publication file to the frozen design, execution binding, external reports, models, and predictions.
