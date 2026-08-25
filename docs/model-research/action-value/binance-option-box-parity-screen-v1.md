# Binance Fixed-Payoff Option Box Screen v1

> **Rejected current snapshot. No edge, profitability, or trading-authority
> claim.** Seven ticker candidates all lacked executable fresh four-leg depth.

This is a second, distinct direction-independent option screen. It consumes the
already source-bound vertical/convexity snapshot and does not refetch the full
contract catalog or all-symbol ticker. Binance Academy documents Binance
options as [European-style and cash-settled][option-mechanics], so a same-expiry
box has a fixed settlement cashflow independent of the underlying price.

For `K1 < K2`, one unit of a long box is:

- buy the `K1` call;
- sell the `K2` call;
- buy the `K2` put;
- sell the `K1` put.

Its expiry cashflow is `K2-K1`. The short box reverses every leg and owes that
same fixed amount. The implementation scales all four legs to the smallest
common exact `LOT_SIZE` quantity.

## Interpretation Gate

A short-box ticker credit greater than its fixed liability is a strict gross
candidate. A long box costing less than its fixed receivable is only a nominal
financing return: it is not an arbitrage claim without commission, capital
lockup, opportunity cost, margin, and settlement evidence. Neither type is
execution evidence until all four required displayed sides are available at the
exact quantity.

## Source-Bound Result

The input snapshot formed 25 underlying-expiry chains with both calls and puts.
The screen evaluated 13,344 strike pairs. Of these, 5,637 long boxes and 8,917
short boxes had all four required ticker prices.

Ticker discovery produced:

- six ETHUSDT short boxes with `0.028`-`0.088` USDT minimum-lot gross surplus
  above the fixed expiry liability;
- one same-day BTCUSDT long box with `0.05` USDT nominal gross carry and a
  579.31% annualized simple rate caused by the very short remaining tenor.

The annualized figure is a mathematical scaling of the ticker-only nominal
surplus, not a yield or profitability claim. The candidate-only depth sweep
requested 18 unique books and found zero executable boxes. Every candidate
lacked at least one required side. The four-leg timestamps also failed the
5,000 ms age or 1,000 ms skew gate; the shared ETH 7,500-call book was more than
81 seconds old in the canonical run.

The screen stopped after that first depth sweep. Exact commission, financing,
margin, inventory, capital lockup, partial-fill risk, and atomic multi-leg
execution were not evaluated because the gross execution gate had already
failed.

The authoritative numeric evidence is
[`binance-option-box-parity-snapshot-v1-2026-08-25.json`](binance-option-box-parity-snapshot-v1-2026-08-25.json),
result SHA-256
`e85b4e270e707c0faa47e3f373d6d345fce448fcbcbcf6a6f39bceec7d9eb229`.

Do not repeat this current-state screen. Reopen it only under a frozen
prospective sampling contract or if authenticated fees, margin, RFQ/atomic
execution, or actual fill evidence materially changes the executable boundary.

[option-mechanics]: https://academy.binance.com/en/articles/what-is-options-trading
