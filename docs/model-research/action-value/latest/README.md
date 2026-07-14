# Round 56: Paired Action Values

> **Rejected development evidence.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 56 trained `24` AMD OpenCL LightGBM artifacts on BTCUSDT, ETHUSDT, and SOLUSDT. Long and short were explicit paired actions. Labels used real one-minute futures paths, gap-through stops, settled funding, and a frozen `16 bps` round-trip stress charge. Model reload error was exactly zero.

| Held-forward metric (Jan-Jun 2024) | Baseline | Governed AI factors |
|---|---:|---:|
| Point MSE skill vs causal constant | 0.413% | 0.535% |
| Pooled Spearman | 0.00542 | 0.01922 |
| Positive-Spearman months | 3/6 | 5/6 |
| Top score quintile, realized stress payoff | -15.41 bps | -14.66 bps |
| q20 pinball skill | 5.950% | 5.912% |
| q20 coverage | 19.68% | 19.70% |

The baseline failed monthly rank consistency and positive top-quintile payoff. The two accepted Fino1 factor programs improved rank consistency to `5/6` positive months and pooled Spearman to `0.01922`, but the top quintile still lost `14.66 bps` per action row after stress costs. The AI treatment therefore failed before economic replay. Trades, ROI, drawdown, and leverage were not evaluated.

The run used `24,096` hourly timestamps and `144,576` paired action rows derived from real minute paths. It generated no synthetic rows and did not read October 2024 or later observations. The percentile analysis is explicitly post-hoc and cannot select a policy.

## Evidence

| View | Graph | Source |
|---|---|---|
| Forecast skill | [SVG](charts/predictive-skill.svg) | [CSV](predictive-summary.csv) |
| Monthly rank | [SVG](charts/monthly-rank.svg) | [CSV](monthly-rank.csv) |
| Payoff stratification | [SVG](charts/payoff-stratification.svg) | [CSV](predictive-summary.csv) |
| Extreme-score diagnosis | [SVG](charts/score-percentiles.svg) | [CSV](score-percentiles.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`model-fold-skill.csv`, `decomposition.csv`, `gates.csv`, `ai-factors.csv`, `failure-analysis.json`, and `screen.json` preserve the remaining evidence. Every chart is regenerated from tracked tabular data.
