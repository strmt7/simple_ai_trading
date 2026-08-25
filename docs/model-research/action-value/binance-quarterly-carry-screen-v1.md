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
