# Integrations Plan

Simple AI Trading now targets multi-asset, high-liquidity day trading on Binance testnet or Demo Trading. This replaces the old single-symbol spot-only integration plan.

## Active Integrations

- Binance `exchangeInfo`: prove symbol status and filters.
- Binance `ticker/24hr`: measure per-symbol quote volume and trade count.
- Binance `ticker/bookTicker`: measure per-symbol bid/ask spread.
- Binance klines: fetch historical bars for training, backtesting, and replay.
- Binance USD-M futures leverage endpoints: read exchange brackets, then apply the app-level `10x` cap.
- DirectML / `torch-directml`: Windows GPU acceleration for AMD, NVIDIA, and Intel GPUs.

## Planned Integrations

- Depth snapshots and diff streams for order-book replay.
- AggTrades replay for more realistic market impact and queue-position models.
- User-data stream reconciliation for signed testnet/demo order state.
- ONNX/Windows ML packaging for future local inference.
- Asset-specific news/sentiment providers beyond broad crypto macro feeds.

## Out Of Scope Until Reconciliation Is Complete

- Mainnet signed execution.
- Autonomous authenticated exchange-order execution.
- Margin/staking/convert products.
- Static symbol allowlists.

All future integrations must preserve CLI/Windows app parity and fail closed when liquidity, position state, or account state cannot be verified.
