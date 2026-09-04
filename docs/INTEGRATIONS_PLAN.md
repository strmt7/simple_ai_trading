# Integrations Plan

Execution remains restricted to BTC, ETH, and SOL on Binance paper, testnet or
Demo, with the independent Polymarket boundary disabled by default. That does
not prohibit public unauthenticated structural-edge research on other assets
or market types. Such research has no account, order, funding or execution
authority. The earlier blanket ban on non-major research was inconsistent with
the current registry and `AGENTS.md`.

The September 4 whole-product reassessment is in
`docs/review/2026-09-04/product-reassessment.md`. Integration priority is
mechanism-specific after-cost evidence and operator visibility, not adding
providers or model features without an economic question.

## Active Integrations

- Binance `exchangeInfo`: prove symbol status and filters.
- Binance `ticker/24hr`: measure per-symbol quote volume and trade count.
- Binance `ticker/bookTicker`: measure per-symbol bid/ask spread.
- Binance klines: fetch historical bars for training, backtesting, and replay.
- Official Binance `aggTrades` archives: construct checksummed one-second futures candles and trade-tape evidence.
- Official Binance `bookTicker` archives: supply exact 100 ms best-bid/offer paths for execution replay.
- Binance USD-M futures leverage endpoints: read exchange brackets, then apply the app-level `20x` cap.
- Signed Binance testnet/demo account endpoints: reconcile bot-owned orders, fills, balances, and positions before opens or closes.

## Planned Integrations

- Event-level depth diff streams for queue-aware order-book research; sampled aggregate-depth archives are insufficient for queue position or maker-fill claims.
- User-data stream reconciliation as a lower-latency complement to the signed REST reconciliation path.
- ONNX Runtime DirectML / Windows ML execution after model export, numerical
  parity, provider discovery, and fault-isolation tests pass.
- Asset-specific news/sentiment providers beyond broad crypto macro feeds.

## Out Of Scope Until Reconciliation Is Complete

- Mainnet signed execution.
- Execution support outside the hard BTC/ETH/SOL base-asset scope.
- Account/state-changing margin, staking or convert operations (public read-only
  mechanism research remains in scope).
- Unverified manual or third-party positions; only provably bot-owned exposure may be managed.

All future integrations must preserve CLI/Windows app parity and fail closed when liquidity, position state, or account state cannot be verified.
