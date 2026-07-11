# Integrations Plan

Simple AI Trading now targets BTC, ETH, and SOL high-liquidity day trading on Binance testnet or Demo Trading. This replaces the old single-symbol spot-only integration plan and rejects non-major assets before data sync, research, or execution.

## Active Integrations

- Binance `exchangeInfo`: prove symbol status and filters.
- Binance `ticker/24hr`: measure per-symbol quote volume and trade count.
- Binance `ticker/bookTicker`: measure per-symbol bid/ask spread.
- Binance klines: fetch historical bars for training, backtesting, and replay.
- Official Binance `aggTrades` archives: construct checksummed one-second futures candles and trade-tape evidence.
- Official Binance `bookTicker` archives: supply exact 100 ms best-bid/offer paths for execution replay.
- Binance USD-M futures leverage endpoints: read exchange brackets, then apply the app-level `20x` cap.
- Signed Binance testnet/demo account endpoints: reconcile bot-owned orders, fills, balances, and positions before opens or closes.
- DirectML / `torch-directml`: Windows GPU acceleration for AMD, NVIDIA, and Intel GPUs.

## Planned Integrations

- Event-level depth diff streams for queue-aware order-book research; sampled aggregate-depth archives are insufficient for queue position or maker-fill claims.
- User-data stream reconciliation as a lower-latency complement to the signed REST reconciliation path.
- ONNX/Windows ML packaging for future local inference.
- Asset-specific news/sentiment providers beyond broad crypto macro feeds.

## Out Of Scope Until Reconciliation Is Complete

- Mainnet signed execution.
- Execution support outside the hard BTC/ETH/SOL base-asset scope.
- Margin/staking/convert products.
- Unverified manual or third-party positions; only provably bot-owned exposure may be managed.

All future integrations must preserve CLI/Windows app parity and fail closed when liquidity, position state, or account state cannot be verified.
