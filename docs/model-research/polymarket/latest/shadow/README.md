# Round 14 one-hour shadow

> **Rejected. Counterfactual only. No orders, fills, profitability claim, or trading authority.**

![Cumulative shadow P&L](cumulative-pnl.svg)

The frozen first-candidate policy won
`3` of
`12` resolved events. Counterfactual net
P&L after observed displayed-depth costs was
`-9.87720` quote. Profit factor was
`0.4557` and maximum drawdown was
`11.69847` quote. Condition-balanced row
accuracy was `0.6205`, but it did not
translate into an executable event-selection edge.

The run observed `91` prediction
rows and `10` opportunity errors. It
submitted zero real orders and observed zero real fills.

## Audit

- [Event-level source table](event-outcomes.csv)
- [Publication integrity](publication-integrity.json)
- [Hash-bound evaluation](../../evidence/round-014-btc-5m-shadow-hour-evaluation-v1.json)
- [Exact source log](../../evidence/round-014-btc-5m-shadow-hour-1785350261587.jsonl)

Regenerate with `python tools/publish_polymarket_shadow_hour.py`.
