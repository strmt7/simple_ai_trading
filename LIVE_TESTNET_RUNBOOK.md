# Non-Mainnet Day-Trading Runbook

This runbook is for Binance testnet or Demo Trading only.

## Preflight

```powershell
simple-ai-trading compute
simple-ai-trading ai
simple-ai-trading status
simple-ai-trading universe
simple-ai-trading risk --paper
simple-ai-trading audit
```

Required state:

- `testnet=true` or `demo=true`
- `dry_run=true` for paper mode
- DirectML/GPU active if AI is enabled
- CPU-only mode accepted only with AI disabled
- at least `min_diversified_assets` eligible symbols from `universe`
- leverage <= `10x`
- reinvest profits disabled unless operator explicitly accepted the warning

## Prepare

```powershell
simple-ai-trading fetch --symbol BTCUSDC --limit 1000
simple-ai-trading train --preset balanced --compute-backend directml
simple-ai-trading evaluate
simple-ai-trading backtest --compute-backend directml
simple-ai-trading backtest-chart
```

Repeat data/training/backtest workflows for each eligible symbol or use the panel workflow as it expands.

## Autonomous Paper Run

```powershell
simple-ai-trading autonomous start --paper
simple-ai-trading autonomous status
simple-ai-trading autonomous stop
```

`autonomous stop` writes `STOPPING` and closes locally tracked open autonomous positions. If no current quote is available, it uses entry price to prevent stale local ledger exposure.

## Blockers

Do not proceed when any of these are true:

- `universe` cannot prove enough liquid symbols.
- `risk` reports a block.
- model/audit checks fail.
- DirectML/GPU is unavailable while AI is enabled.
- leverage request exceeds `10x`.
- signed execution is pointed at mainnet.
- local open positions do not reconcile with the exchange account.

## Notes

Authenticated autonomous exchange-order execution is intentionally disabled until exchange reconciliation and order-close recovery are fully implemented and tested. Use paper mode and signed testnet/demo smoke tests only.
