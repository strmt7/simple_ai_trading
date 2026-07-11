# Implementation Planning: Simple AI Trading

Current direction:

- Windows-first autonomous day-trading CLI and desktop app.
- Binance testnet or Demo Trading first; mainnet signed execution remains blocked by default.
- BTC/ETH/SOL-only by design, with automatic venue and liquidity eligibility inside that hard base-asset scope.
- Conservative default risk profile.
- `20x` maximum app-level futures leverage cap, regardless of exchange-reported maximums.
- DirectML default GPU backend on Windows; CPU-only is allowed but disables AI and warns about slower training/backtesting.

## Product Goals

1. Keep CLI and Windows app functionally aligned through a shared command contract.
2. Require automated liquidity and diversification checks before autonomous trading.
3. Model live-market frictions in backtests: spread, latency, liquidity haircut, market impact, fees, and drawdown stops.
4. Keep stop controls fail-closed for the local ledger so stale autonomous positions do not survive operator stop.
5. Keep all signed execution non-mainnet until exchange reconciliation, order close, and stale-position recovery are fully verified.

## Guardrails

- Default profile: `conservative`.
- Leverage defaults: `5x` conservative, `10x` regular, and `15x` aggressive in futures mode; spot remains `1x`.
- Profit reinvestment: disabled by default, warning when enabled.
- AI: enabled by default only when GPU backend is active; CPU-only disables AI.
- Universe selection: hard BTC/ETH/SOL base-asset scope, then dynamic exchange status, quote volume, trade count, spread, structural-risk, and liquidity checks.
- Backtesting: pessimistic fills, not best-case fills.

## Near-Term Work

- Expand verified historical storage and panel backtests across the supported BTC, ETH, and SOL spot/futures symbols.
- Extend authenticated testnet/demo futures reconciliation with longer interruption and recovery soaks before any mainnet consideration.
- Add ONNX/Windows ML inference packaging once model export stabilizes.
- Expand asset-specific external signal adapters beyond broad crypto/Bitcoin macro feeds.
