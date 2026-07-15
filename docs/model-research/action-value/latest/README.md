# Round 58: Two-Sided Maker Feasibility

> **Rejected post-hoc structural diagnostic.** This is consumed development evidence, not a pre-registered profitability test. It grants no trading, testnet, live, leverage, AI-uplift, or performance authority.

Round 58 asked a narrow question before spending more GPU time: can simultaneous best-bid and best-ask orders complete often enough, and capture enough observed spread, to justify training a symmetric touch-making model? The value-blind probe used checksum-verified official Binance USD-M BTCUSDT, ETHUSDT, and SOLUSDT events from **2023-06-01 UTC**. It read no returns, costs, P&L, strategy outcomes, or policy thresholds.

The answer is no under the prior frozen research cost reference. Two-sided fills occurred in only 2.36-3.18% of eligible decisions, while one-sided fills occurred in 28.03-47.19%. Every symbol's 99th-percentile two-fill placement spread was below 1 bps. The earlier frozen contract models 2 bps maker fee per side, or 4 bps round trip before its additional 1 bps per-side slippage reference. Account-specific production fees must still be queried from the venue; these values are a pinned research comparison, not a universal Binance fee claim.

| Joint fill support | Eligible decisions | Both sides | One side only | No fill |
|---|---:|---:|---:|---:|
| BTCUSDT | 8,639 | 3.07% | 47.19% | 49.74% |
| ETHUSDT | 8,639 | 2.36% | 46.78% | 50.86% |
| SOLUSDT | 8,431 | 3.18% | 28.03% | 68.79% |

| Two-fill placement spread (bps) | p50 | p90 | p99 | Maximum |
|---|---:|---:|---:|---:|
| BTCUSDT | 0.0372 | 0.0374 | 0.0375 | 1.5718 |
| ETHUSDT | 0.0537 | 0.0540 | 0.1589 | 0.7059 |
| SOLUSDT | 0.4835 | 0.4881 | 0.9714 | 7.3206 |

No model was trained, no trades were replayed, and ROI, drawdown, profit factor, leverage, and AI uplift were not computed. This early rejection is intentional: a model cannot manufacture gross spread that is absent from the observed mechanism. The next candidate must use a structurally different source of edge and a newly frozen design.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Joint fill support | [SVG](charts/joint-fill-support.svg) | [CSV](joint-fill-support.csv) |
| Spread versus prior cost reference | [SVG](charts/spread-feasibility.svg) | [CSV](spread-feasibility.csv) |
| Inventory exposure duration | [SVG](charts/inventory-exposure.svg) | [CSV](inventory-exposure.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `fill-bucket-cross.csv`, `failure-analysis.json`, and the exact `screen.json` preserve the underlying evidence. Every chart is regenerated from a tracked CSV.

## Research basis

- [Binance official public market-data archives](https://data.binance.vision/)
- [Avellaneda and Stoikov: High-frequency trading in a limit order book](https://doi.org/10.1080/14697680701381228)
- [Huang, Lehalle, and Rosenbaum: the queue-reactive model](https://arxiv.org/abs/1312.0563)
- [The Market Maker's Dilemma: fill probability and post-fill returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5074873)
