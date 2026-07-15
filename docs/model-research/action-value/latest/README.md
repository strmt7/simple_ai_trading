# Round 59: Funding-Persistence Feasibility

> **Rejected consumed structural evidence.** No profitability, basis-neutrality, AI-uplift, leverage, testnet, live-trading, or execution claim is made.

Round 59 tested whether causally observed positive BTCUSDT, ETHUSDT, and SOLUSDT funding persisted strongly enough to justify downloading the missing synchronized spot history. The runner reconstructed and re-hashed every monthly funding row against 129 checksum-certified Binance archive streams from **December 2021 through June 2025**. It read no price, premium-index, spot, basis, P&L, model, or AI rows.

Positive funding usually remained positive at the next settlement, but ordinary positive-funding episodes did not clear four-leg costs. The rare `>=2` bps trigger produced positive mean seven-day carry after the 32 bps stress reference, yet only 20 BTC, 20 ETH, and 25 SOL non-overlapping episodes existed versus 40 required. BTC and SOL lower 95% mean bounds remained below zero. All 27 symbol cells and all nine BTC/ETH/SOL breadth cells failed, so spot-history ingestion is not authorized.

| Seven-day evidence | P(next positive \| positive) | Ordinary trigger gross mean (bps) | `>=2` bps episodes | `>=2` mean after 32 bps | Median after 32 bps | Lower 95% mean after 32 bps |
|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 91.09% | +14.84 | 20 | +11.44 | +1.57 | -1.80 |
| ETHUSDT | 90.38% | +14.18 | 20 | +14.82 | +4.09 | +2.02 |
| SOLUSDT | 85.72% | -5.97 | 25 | +16.48 | +8.10 | -1.26 |

The 4/28/32 bps values are pinned research references, not account-specific realized fees. The 28 bps reference is two spot fills at 10 bps plus two futures fills at 4 bps; the 32 bps stress reference adds 1 bps to each of four fills. A production path must query both signed commission endpoints and still model synchronized spreads, depth, legging latency, basis change, margin, and liquidation.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Seven-day gross carry versus references | [SVG](charts/seven-day-gross-carry.svg) | [CSV](funding-cells.csv) |
| Elevated-funding confidence and support | [SVG](charts/elevated-funding-support.svg) | [CSV](funding-cells.csv) |
| Funding-sign persistence | [SVG](charts/funding-sign-persistence.svg) | [CSV](sign-persistence.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `breadth-gates.csv`, `source-certificate.json`, `failure-analysis.json`, and the exact `screen.json` preserve the remaining source-bound evidence. Every chart is regenerated from a tracked CSV.

## Research basis

- [Binance USD-M funding history and interval metadata](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-history)
- [Binance spot account commissions](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/account#query-commission-rates-user_data)
- [Binance USD-M user commissions](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account#user-commission-rate)
- [Fundamentals of Perpetual Futures](https://arxiv.org/abs/2212.06888)
