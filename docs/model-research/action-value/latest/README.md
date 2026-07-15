# Round 61: Matched Spot-Perpetual Economic Replay

> **Rejected. No profitability or trading claim.** The elevated-funding seven-day carry family failed executable-capacity and after-cost economic gates. Tick replay, model training, AI evaluation, leverage, testnet, and live trading remain unauthorized.

Round 61 matched a long spot leg with a 1x short USD-M perpetual leg at the same base quantity. It used official checksum-verified Binance minute bars and settled funding, adverse minute high/low execution bounds, actual fill notionals, four taker fees, one extra basis point per fill, and a maximum 1% share of same-side one-minute taker flow. No missing price was interpolated or filled.

| Symbol | Source eligible | Capacity eligible | Mean net bps | Median net bps | Positive | Lower 95% mean | Max drawdown bps | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BTCUSDT | 72/72 (100.00%) | 30 (41.67%) | +1.94 | -6.70 | +43.33% | -7.43 | +123.59 | Rejected |
| ETHUSDT | 76/76 (100.00%) | 20 (26.32%) | +1.54 | -5.56 | +50.00% | -11.13 | +174.00 | Rejected |
| SOLUSDT | 61/62 (98.39%) | 0 (0.00%) | n/a | n/a | n/a | n/a | n/a | Rejected |

Capacity is the first decisive failure: only 30 BTC, 20 ETH, and zero SOL episodes could support every modeled fill. On those admitted BTC/ETH subsets, median stress-net returns and bootstrap lower means were still negative. Positive mean values were not sufficient to pass the precommitted distribution, tail, year-stability, concentration, and breadth gates.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Source and executable-capacity support | [SVG](charts/source-capacity-eligibility.svg) | [CSV](summary.csv) |
| After-cost stress economics | [SVG](charts/stress-net-economics.svg) | [CSV](summary.csv) |
| Mean P&L decomposition | [SVG](charts/pnl-decomposition.svg) | [CSV](episodes.csv) |
| Sequential event-time path | [SVG](charts/cumulative-stress-net.svg) | [CSV](cumulative.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

capacity.csv, yearly.csv, gates.csv, decision-analysis.json, design.json, event-manifest.json, source-certificate.json, and the exact screen.json preserve the remaining source-bound evidence. Every graph is regenerated from tracked numeric data.

## Limits

- Minute extremes are conservative bounds, not historical order-book fills.
- Same-side taker flow is a capacity proxy, not displayed depth.
- The event set is consumed development evidence.
- This seven-day carry screen is separate from the platform's intraday directional day-trading objective.
