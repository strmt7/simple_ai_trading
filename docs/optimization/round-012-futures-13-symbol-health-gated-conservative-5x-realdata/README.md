# Round 012 - Futures 13-Symbol Conservative 5x Real-Data Evidence

Generated on 2026-07-05 from the local exchange-sourced market database. This round used Binance futures 1-minute data, the conservative objective, DirectML-required training/scoring, and 13 automatically selected data-health-gated symbols.

## Result

Round 012 is a failed optimization round. It completed all 13 symbols without CPU fallback, but accepted 0 symbols because every holdout produced fewer than 5 closed trades and no realized profit.

This is intentionally fail-closed evidence. The no-trade result must not be represented as a profitable strategy or used for live trading.

## Portfolio Summary

| Metric | Value |
|---|---:|
| Symbols completed | 13 |
| Accepted symbols | 0 |
| Mean strategy ROI | 0.00% |
| Median strategy ROI | 0.00% |
| Mean passive baseline ROI | -30.22% |
| Worst strategy max drawdown | 0.00% |
| Total closed trades | 0 |
| Mean low-liquidity sample rate | 20.92% |
| Training/scoring backend | DirectML (`privateuseone:0`) |

## Artifacts

- `data/backtest-metrics.csv` contains per-symbol financial metrics and artifact paths.
- `data/report.json` contains the machine-readable provenance, tracked artifact list, and portfolio summary.
- `data/*-timeline.csv.gz` and `data/portfolio-timeline.csv.gz` contain compressed full-resolution graph data.
- `charts/*.svg` contains per-symbol equity-vs-passive-baseline charts rendered from the real holdout data.

## Holdout Windows

| Symbol | Holdout start | Holdout end | Years | Rows | Strategy ROI | Passive ROI | Closed trades | Gate reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETHUSDT | 2024-11-14T18:06:59Z | 2026-06-30T23:51:59Z | 1.62 | 854,266 | 0.00% | -25.10% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| 1000PEPEUSDT | 2025-09-15T22:14:59Z | 2026-06-30T23:51:59Z | 0.79 | 414,818 | 0.00% | -39.21% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| 1000BONKUSDT | 2025-11-05T03:36:59Z | 2026-06-30T23:51:59Z | 0.65 | 342,496 | 0.00% | -32.73% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| JTOUSDT | 2025-11-09T02:36:59Z | 2026-06-30T23:51:59Z | 0.64 | 336,796 | 0.00% | -4.37% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| INJUSDT | 2025-07-12T12:47:59Z | 2026-06-30T23:51:59Z | 0.97 | 508,985 | 0.00% | -31.87% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| IDUSDT | 2025-09-05T03:44:59Z | 2026-06-30T23:51:59Z | 0.82 | 430,328 | 0.00% | -38.88% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| CRVUSDT | 2025-01-14T19:51:59Z | 2026-06-30T23:51:59Z | 1.46 | 766,321 | 0.00% | -39.09% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| EGLDUSDT | 2025-01-18T01:51:59Z | 2026-06-30T23:51:59Z | 1.45 | 761,641 | 0.00% | -46.64% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| 1000LUNCUSDT | 2025-07-18T09:21:59Z | 2026-06-30T23:51:59Z | 0.95 | 500,551 | 0.00% | -5.19% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| ARUSDT | 2025-04-22T18:58:59Z | 2026-06-30T23:51:59Z | 1.19 | 625,254 | 0.00% | -33.22% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| TURBOUSDT | 2025-12-22T15:58:59Z | 2026-06-30T23:51:59Z | 0.52 | 274,074 | 0.00% | -27.09% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| JASMYUSDT | 2025-06-12T18:58:59Z | 2026-06-30T23:51:59Z | 1.05 | 551,814 | 0.00% | -34.86% | 0 | `closed_trades<5; realized_pnl<=0.0` |
| ETHWUSDT | 2025-11-06T15:14:59Z | 2026-06-30T23:51:59Z | 0.65 | 340,358 | 0.00% | -34.57% | 0 | `closed_trades<5; realized_pnl<=0.0` |

