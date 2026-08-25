# Binance Quarterly Cash-and-Carry Screen v1

> **Unqualified current snapshot. No accepted edge, profitability, or trading
> authority.** Displayed basis was positive, but exact account costs and the
> delivery-index/spot-exit relationship remain unresolved.

This screen evaluates a direction-neutral basis trade: buy spot and short the
same base quantity of a USD-M quarterly future. It does not predict whether BTC
or ETH rises or falls. The tool fetched Binance's futures catalog once and then
fetched one spot book and one futures book per selected contract. Each parsed
book was reused for all three frozen quantity tiers.

The arithmetic walks displayed spot asks and futures bids without extrapolating
missing depth. For spot cost `S`, futures sale value `F`, and a stated
sensitivity hurdle `h` in basis points:

```text
gross profit = F - S
after-hurdle profit = F - S - S * h / 10,000
```

The frozen `35` bps hurdle is not an inferred commission or a claim about any
account. It exists only to show how much displayed basis remains after that
explicit deduction.

## Source-Bound Result

The public snapshot covered current- and next-quarter BTCUSDT and ETHUSDT
contracts at 12 exact quantity/contract combinations. All 12 had positive gross
basis, and 9 remained positive after the 35 bps sensitivity hurdle:

| Contract | Quantity range | Gross basis range | After-hurdle range |
| --- | ---: | ---: | ---: |
| BTCUSDT 2026-09-25 | 0.001-0.1 BTC | 41.4190-41.4309 bps | 6.4190-6.4309 bps |
| ETHUSDT 2026-09-25 | 0.01-1 ETH | 21.1733-21.7913 bps | -13.8267--13.2087 bps |
| BTCUSDT 2026-12-25 | 0.001-0.1 BTC | 143.7496-145.1571 bps | 108.7496-110.1571 bps |
| ETHUSDT 2026-12-25 | 0.01-1 ETH | 83.5353-86.4854 bps | 48.5353-51.4854 bps |

Three of the four contracts had at least one fresh after-hurdle-positive size.
The annualized values in the artifact are simple mathematical scalings over the
remaining tenor; they are not yields or profitability claims.

The authoritative evidence is
[`binance-quarterly-carry-snapshot-v1-2026-08-25.json`](binance-quarterly-carry-snapshot-v1-2026-08-25.json),
result SHA-256
`9c9f75565128cd62372ad1971bab09d910583e27e5c47d8eeaeda4e9177b99a2`.
Its test reconstructs the artifact hash, implementation hashes, and every
quantity result from the embedded books.

## Why This Is Not Yet an Accepted Edge

The following terms are not established by public depth:

- authenticated spot and futures commission for the intended account;
- any delivery or settlement charge;
- capital opportunity cost and collateral haircuts;
- liquidation buffer and the cost of keeping the short future solvent;
- the futures delivery-index value versus the realizable spot disposal price;
- spot disposal depth, outages, taxes, and operational failure modes;
- persistence before discovery and capacity through time.

The spot REST response has no exchange event timestamp, so close HTTP receipt
times are not proof of an atomic two-leg quote. The future's delivery value also
must not be treated as identical to a spot exit without evidence.

Do not repeat this snapshot merely because the basis moves. Reopen the study
only with a frozen prospective sampling contract or materially new exact fee,
collateral, settlement, execution, or fill evidence. No order was placed and no
credential was used.

## Frozen Account-Fee Gate

Read-only mainnet account-evidence authority was received on 2026-08-25. The
exact capture is frozen in
[`binance-quarterly-carry-account-evidence-contract-v1.json`](binance-quarterly-carry-account-evidence-contract-v1.json),
result SHA-256
`901c16bf3e7e4082339f3ddd2a910a904a3cd46d51c0dc16f7074c16351145e5`.
It permits only seven signed GETs: spot commission for BTCUSDT and ETHUSDT,
quarterly-futures commission for the four source-snapshot contracts, and the
minimal futures account configuration. Each signed request has a fresh exchange
clock request, zero retries, and a durable self-hashed before/after journal.
Balances, positions, API keys, secrets, signatures, and signed URLs are not
persisted. The BNB discount is retained but cannot be applied because the fee
payment asset and sufficient BNB balance are not proved.

The required process variables were absent when the contract was frozen, so no
authenticated request was attempted and no attempt was consumed. Do not recover
secrets from chat, shell history, logs, or repository files and do not ask for
them to be pasted. Once both documented environment variables are available,
run the frozen capture once. Apply its exact buyer/seller and maker/taker
components only as a non-synchronous overlay on this snapshot. That overlay can
reject contracts on fees, but it cannot accept a current edge or justify fresh
books until settlement, collateral, liquidation, exit-basis, persistence, and
capacity gates are separately frozen.

## Delivery-Basis Adjudication

A separately frozen historical audit attempted to compare the latest eight
completed BTCUSDT and ETHUSDT USD-M quarterly delivery prices with post-delivery
spot bars. Its result is retained only for provenance. All 16 historical
`deliveryTime` fields were at 00:00 UTC, while every quarterly `deliveryDate` in
the independently captured current exchange catalog was at 08:00 UTC. The audit
treated the historical field as an exact spot-window epoch without proving that
semantic equivalence. Its mismatch values and hold-to-delivery rejection are
therefore invalid.

The source-bound timestamp adjudication neither resamples nor assumes that
adding eight hours is correct. Binance's official quarterly-delivery rule now
binds the normal schedule to the last Friday at 08:00 UTC, while explicitly
allowing postponement under extreme conditions. The timing artifact is
[`binance-quarterly-delivery-time-semantics-v1.json`](binance-quarterly-delivery-time-semantics-v1.json),
result SHA-256
`2a52b558f8bc1332cbf2deb41c4e8d4f01bf44d4276ebcc901b3768d4d8516db`.

A separately frozen 16-contract pre-delivery unwind contract must validate the
actual historical futures cutoff before using any basis observation. Its one
primary 10-minute horizon and adverse futures-high/spot-low arithmetic cannot
accept an edge. It permits 32 public requests with no retry or replacement and
stopped terminally after the first futures/spot request pair. The expired
futures endpoint returned 70 rows through 08:09 UTC, including ten flat rows
at/after the scheduled 08:00 delivery with zero volume and zero trades. That
violated the exact-60/no-later-bar cutoff gate. The 60 pre-delivery rows may not
be salvaged and the audit may not be rerun. Returned-kline presence is not an
authenticated order-state test; no historical basis observation was accepted.
Exact account fees, margin, liquidation, capital cost, and executable fills
remain unresolved. The contract is
[`binance-quarterly-pre-delivery-unwind-contract-v1.json`](binance-quarterly-pre-delivery-unwind-contract-v1.json),
result SHA-256
`f61a8c9dfd86274292c5dae154120871ea5358e2a5ca004b92574e6bdcb7657c`.
The terminal audit is
[`binance-quarterly-pre-delivery-unwind-audit-v1-2026-08-25.json`](binance-quarterly-pre-delivery-unwind-audit-v1-2026-08-25.json),
result SHA-256
`07556c4c128fdde32b8bc3ade55134e25eedec157715585aac9e561d87ac9e5a`.
Its source-bound adjudication is
[`binance-quarterly-pre-delivery-unwind-terminal-adjudication-v1.json`](binance-quarterly-pre-delivery-unwind-terminal-adjudication-v1.json),
result SHA-256
`e45df8dbffdb8e8e09a542ad3cf2f2f7fe855a775c10f9c07cfa30b290505521`.

The original invalid audit is
[`binance-quarterly-delivery-basis-audit-v1-2026-08-25.json`](binance-quarterly-delivery-basis-audit-v1-2026-08-25.json),
result SHA-256
`5476fdb43a24bd2d3a31c10321de968f63fd33eca20603e4891fa8d838a134a4`.
The authoritative correction is
[`binance-quarterly-delivery-basis-timestamp-adjudication-v1.json`](binance-quarterly-delivery-basis-timestamp-adjudication-v1.json),
result SHA-256
`1f669b24c09917e8b080515e8733ba0adea68e74745e2cfafc9dd8f9a45c7f88`.
