# Binance Non-Mainnet Runbook

This runbook covers paper, Binance testnet, and Demo Trading only. It does not
authorize mainnet or claim that a model is profitable.

## Preflight

```powershell
uv run simple-ai-trading doctor
uv run simple-ai-trading compute
uv run simple-ai-trading ai
uv run simple-ai-trading status --compact
uv run simple-ai-trading universe
uv run simple-ai-trading reconcile
uv run simple-ai-trading risk --paper
uv run simple-ai-trading api-budget --compact
```

Proceed only when the selected venue is non-mainnet, the supported BTC/ETH/SOL
universe passes current liquidity checks, ownership reconciles, risk is clear,
and every known API window is below 80% use. CPU mode requires AI off.

## Prepare Evidence

```powershell
uv run simple-ai-trading data-health --json
uv run simple-ai-trading model-lab --market futures --quote-asset USDT --interval 1s
uv run simple-ai-trading backtest-chart --output data/backtest-performance.svg
```

A command completing successfully is not model promotion. Check the generated
report's source cutoff, gaps, costs, holdout status, trade count, drawdown, and
authority fields.

## Run Paper Mode

```powershell
uv run simple-ai-trading autonomous start --paper
uv run simple-ai-trading autonomous status
uv run simple-ai-trading autonomous pause
uv run simple-ai-trading autonomous stop
uv run simple-ai-trading reconcile
```

Pause blocks new entries. Stop requests venue-specific closure of bot-owned
positions only. It never invents a fill from an entry price or cached mark. If
fresh quotes, ownership, broker acknowledgement, or the active worker are
unavailable, Stop returns a visible failure and the position remains explicitly
open for reconciliation.

## Interruptions

After a process, network, or venue interruption:

1. Block new exposure.
2. Restore connectivity without replaying ambiguous submissions.
3. Reconcile exact bot order IDs, fills, balances, and positions.
4. Refresh market data and API-budget telemetry.
5. Observe the configured cooldown before any new decision.

Do not continue when exchange-only exposure, local-only exposure, quantity
drift, unknown orders, stale books, malformed ledgers, or unresolved fills are
present. Use [Live-market simulation](docs/LIVE_MARKET_SIMULATION.md) for the
full execution and outage contract.
