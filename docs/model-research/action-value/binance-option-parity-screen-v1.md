# Binance Option Payoff-Parity Screen v1

> **Rejected current snapshot. No edge, profitability, or trading-authority
> claim.** Two ticker-only discrepancies were both negative at exact displayed
> depth before fees.

This screen tests payoff identities that do not predict whether BTC, ETH, or
SOL will rise, fall, or remain flat. Binance documents its options API and
contract fields in the [official Options API reference][options-api]. Binance
Academy states that Binance options are [European-style and cash-settled][option-mechanics].

For strikes `K1 < K2 < K3`, the screen evaluates:

- call vertical: buy `K1`, sell `K2`;
- put vertical: buy `K2`, sell `K1`;
- strike convexity: buy `K1` with weight `K3-K2`, sell `K2` with weight
  `K3-K1`, and buy `K3` with weight `K2-K1`.

Each portfolio has a nonnegative expiry payoff for every underlying settlement
price. Strike-gap weights are reduced to primitive integers, then scaled to the
smallest exact quantity satisfying every contract's `LOT_SIZE` step and minimum.
No floating-point arithmetic or inferred lot rule enters the result.

## Fixed Workflow

1. Fetch `exchangeInfo` once and retain only `TRADING`, `CRYPTO_OPTIONS`,
   unit-one contracts for BTCUSDT, ETHUSDT, and SOLUSDT.
2. Fetch the all-symbol ticker once. It is discovery data only because it has
   prices but no displayed quantities.
3. Enumerate every same-underlying, same-expiry, same-side vertical and
   arbitrary-strike convexity identity.
4. Fetch depth only for gross-positive ticker candidates. Reprice the exact
   minimum lot portfolio by walking displayed levels.
5. Require book age no greater than 5,000 ms and cross-leg event-time skew no
   greater than 1,000 ms. Continue to at most three depth sweeps only while a
   candidate remains fresh and gross-positive.
6. Stop on HTTP 429 without an automatic retry. Honor `Retry-After`; never
   refetch the contract catalog within a confirmation sweep.
7. Even a persistent gross credit remains unqualified until authenticated exact
   commission, account margin/short inventory, and atomic multi-leg execution
   are independently verified.

## Source-Bound Result

The canonical snapshot contains 1,538 contracts in 50 chains. It evaluated
26,688 vertical pairs and 338,904 convexity triples; 17,045 verticals and
208,419 triples had the required ticker sides. Two convexity portfolios showed
minimum-lot ticker credits of `0.05` USDT.

The first candidate-only depth sweep rejected both:

| Minimum-lot ticker credit | Displayed-depth gross | Freshness |
|---:|---:|---|
| 0.05 USDT | -0.15 USDT | failed |
| 0.05 USDT | -0.15 USDT | failed |

Because no candidate remained gross-positive, the screen correctly stopped
before fees, margin, atomicity, or another sweep. The authoritative numeric
evidence is
[`binance-option-parity-snapshot-v1-2026-08-25.json`](binance-option-parity-snapshot-v1-2026-08-25.json),
result SHA-256
`ceca2f61ab1da16285190afcb90c276a10b032fb7d264c90656aaf2f7266c253`.

Do not rerun this current-state screen because quote repetition cannot create
evidence of a persistent edge. Reopen the hypothesis only under a frozen
prospective sampling contract or when materially new authenticated fee,
margin, RFQ/atomic-execution, or fill evidence changes the executable boundary.

[options-api]: https://developers.binance.com/en/docs/catalog
[option-mechanics]: https://academy.binance.com/en/articles/what-is-options-trading
