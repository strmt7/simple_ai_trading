# Round 57: Queue-Censored Make/Take

> **Rejected development evidence.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 57 trained fixed three-seed queue-fill and payoff ensembles with AMD OpenCL LightGBM on real official Binance USD-M BTCUSDT, ETHUSDT, and SOLUSDT events. Decisions were spaced 10 seconds apart. Labels used an explicit 750 ms placement delay, a 15-second queue-censored passive-order lifetime, observed spread and queue, a 100 ms path grid, fees, slippage, protection latency, and a five-minute post-fill lifecycle.

The queue-fill mechanism generalized. The directional payoff mechanism did not. Every fill cell passed in policy calibration and consumed evaluation, but only 3/12 evaluation payoff cells passed and every evaluation top-score quintile remained negative after costs. The run therefore stopped before policy selection and economic replay. Trades, ROI, drawdown, leverage, and AI uplift were not evaluated.

| Evaluation fill skill | Long log loss | Long Brier | Short log loss | Short Brier |
|---|---:|---:|---:|---:|
| BTCUSDT | 19.63% | 22.40% | 19.33% | 21.80% |
| ETHUSDT | 18.02% | 19.10% | 18.24% | 19.19% |
| SOLUSDT | 6.03% | 5.22% | 6.57% | 5.01% |

| Top-quintile realized net payoff (bps) | Passive long | Passive short | Aggressive long | Aggressive short |
|---|---:|---:|---:|---:|
| BTCUSDT | -10.02 | -8.76 | -12.15 | -11.33 |
| ETHUSDT | -10.98 | -9.46 | -12.71 | -10.91 |
| SOLUSDT | -7.48 | -16.52 | -16.58 | -10.72 |

The retained hypothesis is narrow: queue-fill survival is useful execution infrastructure. The rejected hypothesis is that this L1/tape directional model can clear the frozen maker-entry/taker-exit or taker-entry/taker-exit costs. No threshold, leverage, or language model is allowed to repair that negative mechanism on consumed outcomes.

## Evidence

| View | Graph | Source |
|---|---|---|
| Queue-fill proper-score skill | [SVG](charts/fill-survival-skill.svg) | [CSV](fill-survival.csv) |
| Realized top-quintile payoff | [SVG](charts/top-quintile-net-payoff.svg) | [CSV](conditional-payoff.csv) |
| Opportunity-weighted action value | [SVG](charts/expected-action-value.svg) | [CSV](action-values.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `model-artifacts.csv`, `gates.csv`, `failure-analysis.json`, and `screen.json` preserve the remaining source-bound evidence. Every chart is regenerated from tracked tabular data.
