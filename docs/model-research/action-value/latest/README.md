# Round 52: Executable-Support Hurdle

> **Rejected consumed-development screen.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 52 corrected the measured Round 51 target-policy mismatch: training, early stopping, calibration, thresholds, scoring, and replay now use one hash-bound side-specific executable predicate. It trained 27 OpenCL LightGBM models on verified Binance USD-M BTC, ETH, and SOL tick data and reused the sealed causal FinCast feature matrices.

The correction restored activity, but calibration rejected every policy. The deterministic hurdle produced 9 calibration trades at `-1.057757` base and `-2.391091` stressed bps/trade. The later consumed interval showed 15 trades at `+5.162619` base and `+3.807622` stressed bps/trade, but that reversal was not authorized by calibration and cannot be selected. The direct model remained flat. FinCast produced 3 negative calibration trades and 7 negative consumed-evaluation trades.

Profitable-event classification improved over training prevalence, while expected-payoff magnitude did not: mean evaluation expected-payoff MSE skill was negative for every architecture. FinCast improved average probability log loss by `0.001463` and expected-payoff Spearman by `0.000375`, below both frozen `0.005` gates.

## Evidence

| View | Graph | Source |
|---|---|---|
| Executable support | [SVG](charts/executable-support.svg) | [CSV](support.csv) |
| Forecast quality | [SVG](charts/forecast-quality.svg) | [CSV](forecast.csv) |
| Policy economics | [SVG](charts/policy-economics.svg) | [CSV](policy-grid.csv) |
| Fixed-policy daily path | [SVG](charts/daily-equity.svg) | [CSV](daily-policy.csv) |
| FinCast uplift | [SVG](charts/ai-uplift.svg) | [CSV](ai-uplift.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`screen.json` is the complete source report. `report.json` binds this publication to the frozen design, execution binding, external report, and every published file. Model and prediction artifact hashes are recorded in [models.csv](models.csv) and `screen.json`; flattened gate outcomes are in [gates.csv](gates.csv).
