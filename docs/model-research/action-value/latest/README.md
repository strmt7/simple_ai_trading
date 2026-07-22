# Round 72: Spot-Perpetual Price Discovery

> **Rejected. No profitability or trading claim.** None of the nine BTC, ETH, and SOL symbol-horizon components passed the frozen predictive gate. Terminal data, P&L replay, leverage, testnet, and live trading remain closed.

Round 72 tested whether causal one-second spot flow, perpetual flow, basis, and lead-lag features improved autonomous crypto day-trading forecasts at 30, 60, and 300 seconds. The official Binance corpus contains 17,884,800 one-second rows from one deterministic UTC day per month across October 2020 through June 2026. Six rolling out-of-sample folds covered April 2023 through March 2026; April-June 2026 remained sealed.

All 216 shallow LightGBM models ran through OpenCL and reproduced predictions exactly after serialization. The spot-perpetual layer improved only 15 of 36 primary losses, by amounts far below the frozen hurdle; every adjusted q-value was 0.9695-0.9814. The return-regression head failed to beat a zero-return forecast on MSE for every symbol and horizon.

| Symbol | Horizon | Direction log-loss skill | Return MSE skill vs zero | Spot incremental log-loss | BH q | Decision |
|---|---:|---:|---:|---:|---:|---|
| BTCUSDT | 30s | -0.006% | -0.040% | -0.042% | 0.9814 | Rejected |
| BTCUSDT | 60s | -0.007% | -0.039% | +0.000% | 0.9695 | Rejected |
| BTCUSDT | 300s | +0.109% | -0.000% | +0.022% | 0.9695 | Rejected |
| ETHUSDT | 30s | +0.071% | -0.005% | +0.013% | 0.9695 | Rejected |
| ETHUSDT | 60s | +0.050% | -0.012% | -0.007% | 0.9695 | Rejected |
| ETHUSDT | 300s | +0.003% | -0.046% | -0.011% | 0.9695 | Rejected |
| SOLUSDT | 30s | +0.000% | -0.005% | -0.005% | 0.9695 | Rejected |
| SOLUSDT | 60s | +0.010% | -0.010% | +0.021% | 0.9695 | Rejected |
| SOLUSDT | 300s | -0.020% | -0.142% | -0.010% | 0.9695 | Rejected |

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Directional proper-score skill | [SVG](charts/primary-binary-skill.svg) | [CSV](components.csv) |
| Return skill versus zero | [SVG](charts/primary-continuous-skill.svg) | [CSV](components.csv) |
| Increment from spot and basis features | [SVG](charts/spot-perpetual-increment.svg) | [CSV](components.csv) |
| Day-block confidence | [SVG](charts/day-block-confidence.svg) | [CSV](components.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

The exact [evaluation](evaluation.json), [108-row metric table](metrics.csv), [36 paired comparisons](feature-comparisons.csv), [216 model records](models.csv), corpus ingestion certificate, design, inventory, and implementation freeze are tracked beside the graphs. Every SVG is regenerated from these numeric files.

## Limits

- This is a predictive screen, not an execution or after-cost backtest.
- One full UTC day per month is representative sampling, not continuous tick coverage of every day.
- Binance spot and perpetual crypto trade continuously; UTC dates are sampling blocks, not formal closes. Listed ETFs and futures follow their own venue calendars and were excluded.
- Aggregate trades do not expose historical quotes, queue position, spread, impact, or receive latency.
- The consumed development result cannot be rescued by post hoc tuning. A materially new hypothesis requires a new preregistered round.
