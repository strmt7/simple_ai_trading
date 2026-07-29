# Independent Polymarket Execution

## Status

The BTC-only Polymarket path is live-capable infrastructure, disabled by
default. It has no promoted trading model, live-money release authority,
authenticated account test, or profitability claim. Binance testnet balances,
positions, orders, credentials, and execution state are never shared with it.

Optional Binance BTCUSDT spot and USD-M futures observations are read-only
exogenous features. They can preserve, reduce, or veto a separately approved
Polymarket action; they cannot create one.

```mermaid
flowchart LR
    B["Public BTC price discovery"] --> F["Staleness and sequence gate"]
    P["Polymarket public book"] --> M["Polymarket model and risk"]
    F --> M
    M --> A["Runtime authority latch"]
    U["Authenticated user stream"] --> A
    R["REST ownership reconciliation"] --> A
    A --> E["Exact-ID CLOB V2 gateway"]
    E --> L["Hash-bound SQLite ownership ledger"]
    U --> L
    R --> L
```

## Implemented

- Official `py-clob-client-v2==1.1.0` V2 signing and exact EIP-712 order-hash
  reservation before submission.
- Official `polymarket-client==0.2.0` pinned for the typed transaction and
  redemption path.
- Environment-only, redacted credentials and a dedicated-wallet requirement.
- FAK, FOK, or bounded GTD orders only. Live Polymarket execution is BTC-only;
  every SELL must close confirmed bot-owned inventory.
- Intent TTL, tick alignment, quote-plus-fee ceiling, token inventory ceiling,
  and exact pUSD or outcome-token balance/allowance checks.
- One POST attempt. A transport ambiguity becomes `unknown`; it is never
  blindly retried.
- Exact owned-order cancellation only. Account-wide heartbeat, cancel-all, and
  market-wide cancellation are prohibited.
- Hash-bound order and fill snapshots, an append-only audit chain, exact fill
  economics, cumulative-fill caps, and restart reconciliation.
- Independent user-stream and REST loops. Stream freshness is mandatory for
  opens; a fresh ownership reconciliation can still permit an owned close.
- Foreign orders, positions, or authenticated stream events fail closed and
  are never modified.

## Still Closed

- No Polymarket model has passed prospective, after-cost promotion gates.
- No account credentials are available for an authenticated host test.
- Automatic redemption is not yet connected to the durable transaction
  ledger.
- CLI and Windows controls do not yet expose this boundary.

These gaps block release authority. A public API check or offline signature is
not an authenticated order, cancellation, fill, or redemption test.

## Current Public Host Evidence

On 2026-07-29, the production read endpoints returned CLOB protocol V2 and a
non-blocked CH/ZH geoblock result from this host. Two current BTC five-minute
markets matched Gamma and CLOB identity, token, tick-size, minimum-size, fee,
and payload-hash checks. An offline SDK signature against one current token
preserved the requested 5 shares at 0.50 and derived the expected V2 order
hash. No order was submitted.

## Primary Contracts

- [CLOB V2 migration](https://docs.polymarket.com/v2-migration)
- [Order lifecycle](https://docs.polymarket.com/concepts/order-lifecycle)
- [Order placement](https://docs.polymarket.com/trading/place-orders)
- [Real-time order updates](https://docs.polymarket.com/trading/realtime-order-updates)
- [Wallets and authentication](https://docs.polymarket.com/trading/wallets-auth)
- [Geographic restrictions](https://docs.polymarket.com/api-reference/geoblock)
- [Rate limits](https://docs.polymarket.com/api-reference/rate-limits)

