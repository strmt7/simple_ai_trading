# Round 60: Full-History Funding Replication

> **Structural gate passed; no profitability claim.** This consumed funding-only study authorizes one separately frozen spot-perpetual replay. It does not authorize a model, AI, leverage, testnet, or live trading.

Round 60 kept every Round 59 trigger, horizon, cost reference, bootstrap seed, and breadth threshold unchanged, then expanded to every frozen complete official monthly funding archive: **January 2020-June 2026** for BTC and ETH and **September 2020-June 2026** for SOL. All 226 archives passed Binance SHA-256 sidecar checks, and every database month was re-hashed against its certified row stream.

Exactly one of nine breadth cells passed: a causally observed settled funding rate of at least `2` bps followed by a non-overlapping seven-day window. All three symbols exceeded the precommitted 40-episode, 55% positive, positive-median, and positive lower-95%-mean gates after the 32 bps research reference.

| Seven-day `>=2` bps evidence | Funding rows | P(next positive \| positive) | Episodes | Mean after 32 bps | Median after 32 bps | Positive after 32 bps | Lower 95% mean after 32 bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 7,119 | 92.10% | 72 | +36.02 | +20.13 | 66.67% | +18.48 |
| ETHUSDT | 7,119 | 92.06% | 76 | +51.82 | +34.42 | 78.95% | +29.58 |
| SOLUSDT | 6,424 | 87.13% | 62 | +32.48 | +13.52 | 74.19% | +6.30 |

These are funding-carry references, not trade returns. The replay still has to price both legs and model spot-perpetual basis change, synchronized spread and depth, queueing, market impact, legging latency, account-specific commissions, margin, liquidation, outages, and unwind behavior. No price, basis, P&L, model, or AI row was read in this round.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Seven-day gross funding versus references | [SVG](charts/seven-day-gross-carry.svg) | [CSV](funding-cells.csv) |
| Passing cell confidence and support | [SVG](charts/elevated-funding-support.svg) | [CSV](funding-cells.csv) |
| Round 59-to-60 sample comparison | [SVG](charts/replication-comparison.svg) | [CSV](round59-comparison.csv) |
| Funding-sign persistence | [SVG](charts/funding-sign-persistence.svg) | [CSV](sign-persistence.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `breadth-gates.csv`, `source-certificate.json`, `decision-analysis.json`, and the exact `screen.json` preserve the remaining source-bound evidence. Every graph is regenerated from a tracked table.

## Research basis

- [Binance USD-M funding history and interval metadata](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-history)
- [Binance spot account commissions](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/account#query-commission-rates-user_data)
- [Binance USD-M user commissions](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account#user-commission-rate)
- [Fundamentals of Perpetual Futures](https://arxiv.org/abs/2212.06888)
