# Round 62: Coarse Depth-Stress Transition Forecast

> **Predictive gate passed. No profitability or trading claim.** The result authorizes only a separately frozen, paired after-cost economic replay. It does not authorize orders, leverage, testnet trading, or live trading.

Round 62 evaluated whether causal coarse-depth features improve 60-second and 300-second forecasts of calm, mixed, and stressed depth states for BTCUSDT, ETHUSDT, and SOLUSDT perpetual futures. The source is Binance's checksum-verified `bookDepth` archive. These crypto markets trade continuously; UTC days and months are sampling and evaluation buckets, not formal market closes.

The expanding walk-forward evaluation covered 42 eligible months from January 2023 through June 2026. Its 35 untouched test months ran from August 2023 through June 2026. All 210 shallow LightGBM artifacts used OpenCL, had distinct SHA-256 identities, and retained no trading authority.

| Symbol | Horizon | OOS examples | Accuracy | NLL improvement vs transition | Brier improvement vs transition |
|---|---:|---:|---:|---:|---:|
| BTCUSDT | 60 s | 1,435,196 | 86.89% | 26.09% | 18.34% |
| BTCUSDT | 300 s | 279,825 | 80.95% | 21.29% | 16.30% |
| ETHUSDT | 60 s | 1,436,643 | 83.07% | 23.75% | 17.81% |
| ETHUSDT | 300 s | 280,116 | 76.26% | 18.93% | 15.22% |
| SOLUSDT | 60 s | 1,432,400 | 81.97% | 20.91% | 15.89% |
| SOLUSDT | 300 s | 279,267 | 74.88% | 17.19% | 13.92% |

All 24 preregistered challenger comparisons passed the 0.5% minimum relative-improvement gate and Benjamini-Hochberg FDR at the 10,000-draw permutation floor, `q = 0.000099990001`. Proper scoring rules, rather than accuracy alone, controlled acceptance. The largest absolute difference between forecast and observed stressed-state prevalence was 1.02 percentage points.

## Evidence

- [Exact screen report](round-062-depth-stress-transition-report.json), embedded report SHA-256 `c1a0bf204820715fe45209bfd592d11bce14e8d09b55b8639a0b06aaae44ccba`
- [Comparison table](round-062-depth-stress-comparisons.csv)
- [Aggregate horizon scores](round-062-depth-stress-horizons.csv)
- [Model records](round-062-depth-stress-models.csv)
- [Source bindings](round-062-depth-stress-sources.csv)
- [Corpus ingestion report](round-062-depth-corpus-ingestion-report.json)
- [Frozen design](round-062-depth-stress-transition-design.json)
- [First-execution erratum](round-062-execution-erratum-001.json)

## Limits

The archive contains cumulative percentage-band depth at roughly 30-second cadence. It is not order-by-order L2 data and cannot reconstruct spread, queue position, market impact, or subsecond execution. Predicting a depth state is not predicting return direction or net P&L. A separately frozen replay must still test incremental strategy outcomes after fees, spread, latency, impact, funding, capacity, outages, liquidation controls, and drawdown limits before this signal can enter any risk or execution path.
