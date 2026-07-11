# Round 14: action mapping still failed policy

**Rejected.** Direction x magnitude improved the Huber model over mean-head selection, but its top-500 policy result was **-1.06 bps gross** and **-13.11 bps exact net**. Its later development result improved to **+5.59 bps gross** but remained **-6.46 bps exact net**.

| Model | Best activity-qualified policy mapping | Policy gross | Policy exact net | Development exact net |
| --- | --- | ---: | ---: | ---: |
| MLP Huber/direction | direction x magnitude | -1.06 bps | -13.11 bps | -6.46 bps |
| MLP GMADL + coherence 0.25 | mean | -1.11 bps | -13.17 bps | -6.88 bps |
| LightGBM baseline | direction | -5.63 bps | -17.66 bps | -8.86 bps |


![After-cost mapping results](charts/after-cost-performance.svg)

![Forecast and coherence quality](charts/forecast-quality.svg)

![Action funnel](charts/action-funnel.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, 2023-05-16 through 2023-07-06 UTC; 230,999 causal events from 878,025 exact-BBO rows. The development role is consumed and 2023-07-07 remains untouched. Rows overlap, so these are not trades, an equity curve, ROI, or trading authority.

Data: [candidates.csv](candidates.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
