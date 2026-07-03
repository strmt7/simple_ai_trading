# Implementation Planning: Simple AI Trading

Current direction:

- Windows-first autonomous day-trading CLI and desktop app.
- Binance testnet or Demo Trading first; mainnet signed execution remains blocked by default.
- Multi-asset by design, with automatic high-liquidity universe selection.
- Conservative default risk profile.
- `10x` maximum app-level leverage cap, regardless of exchange-reported maximums.
- DirectML default GPU backend on Windows; CPU-only is allowed but disables AI and warns about slower training/backtesting.

## Product Goals

1. Keep CLI and Windows app functionally aligned through a shared command contract.
2. Require automated liquidity and diversification checks before autonomous trading.
3. Model live-market frictions in backtests: spread, latency, liquidity haircut, market impact, fees, and drawdown stops.
4. Keep stop controls fail-closed for the local ledger so stale autonomous positions do not survive operator stop.
5. Keep all signed execution non-mainnet until exchange reconciliation, order close, and stale-position recovery are fully verified.

## Guardrails

- Default profile: `conservative`.
- Default leverage: `1x`.
- Profit reinvestment: disabled by default, warning when enabled.
- AI: enabled by default only when GPU backend is active; CPU-only disables AI.
- Universe selection: no static allowlist; use exchange status, quote volume, trade count, spread, structural-risk patterns, and liquidity score.
- Backtesting: pessimistic fills, not best-case fills.

## Near-Term Work

- Expand per-symbol historical storage and panel backtests across all eligible symbols.
- Add exchange reconciliation for authenticated futures positions before enabling autonomous live exchange execution.
- Add ONNX/Windows ML inference packaging once model export stabilizes.
- Expand asset-specific external signal adapters beyond broad crypto/Bitcoin macro feeds.
