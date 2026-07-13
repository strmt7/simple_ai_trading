# Round 45: Joint TCN and SAM

> **Beta research warning:** rejected, selection-contaminated development evidence. No model is approved for testnet, live day trading, leverage, or autonomous execution.

Round 45 compared a joint 213-channel cross-asset distributional TCN trained with AdamW and sharpness-aware minimization (SAM). Six three-seed artifacts trained on the AMD GPU through DirectML, reloaded exactly, and emitted zero fallback warnings.

![Forecast quality](charts/forecast-quality.svg)

| Horizon | AdamW skill | AdamW Spearman | SAM skill | SAM Spearman |
|---:|---:|---:|---:|---:|
| 1 h | 3.75% | 0.0275 | 3.80% | 0.0315 |
| 4 h | 3.39% | 0.0368 | 3.34% | 0.0215 |
| 12 h | 3.20% | 0.0665 | 3.07% | 0.0642 |
| 24 h | 1.58% | 0.0445 | 1.50% | 0.0390 |

Both candidates preserved weak forecast information, but joint training made seed agreement materially worse. AdamW reached only `0.189` and SAM `0.181` against the frozen `0.500` floor. SAM therefore did **not** establish an optimizer improvement.

![Seed stability](charts/seed-stability.svg)

![Per-symbol forecast quality](charts/symbol-forecast.svg)

The consensus mapping restored activity: AdamW made `898` trades and SAM `947` across `272` active days. AdamW lost `-35.98%`. SAM's `+22.26%` base point estimate is **not validated**: maximum drawdown was `29.48%`, profit factor `1.029`, only `4/9` months were positive, and the stress bootstrap lower bound was `-1.043` bps/hour.

![Policy economics](charts/policy-economics.svg)

![Monthly economics](charts/monthly-economics.svg)

![Dated equity](charts/daily-equity.svg)

![Research progress](charts/research-progress.svg)

Data: [horizons](horizons.csv) | [symbol horizons](symbol-horizons.csv) | [forecast diagnostics](diagnostics.csv) | [seed stability](seed-stability.csv) | [models](models.csv) | [roles](roles.csv) | [trades](trades.csv) | [replays](replays.csv) | [monthly economics](monthly.csv) | [symbol economics](symbols.csv) | [daily equity](daily-equity.csv) | [sources](sources.csv) | [progress](progress.csv) | [failure analysis](../round-045-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
