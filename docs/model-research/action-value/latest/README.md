# Round 13: gross edge, no taker viability

**Rejected.** The best model produced **+5.44 bps gross**, but only **-6.61 bps after measured taker costs** across its 500 strongest development forecasts.

| Model | AUC | Spearman IC | Top 500 gross | Top 500 exact net |
| --- | ---: | ---: | ---: | ---: |
| MLP + bounded GMADL | 0.571 | 0.111 | +5.44 bps | -6.61 bps |
| MLP + Huber/direction | 0.570 | 0.108 | +5.07 bps | -6.98 bps |
| LightGBM baseline | 0.564 | 0.123 | +3.85 bps | -8.20 bps |


![After-cost performance](charts/after-cost-performance.svg)

![Forecast quality](charts/forecast-quality.svg)

![Promotion funnel](charts/action-funnel.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 230,999 causal events from 878,025 exact-BBO rows. The development window is consumed and the 2023-07-07 terminal day remains untouched. Top-row forecasts overlap, so they are not trades, an equity curve, ROI, or trading authority.

Data: [candidates.csv](candidates.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
