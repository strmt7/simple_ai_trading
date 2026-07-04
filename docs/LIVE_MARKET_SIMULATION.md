# Live-Market Simulation Notes

Simple AI Trading backtests are intentionally pessimistic. A strategy that only works in a frictionless candle replay is not acceptable for autonomous day trading.

## Research Baseline

Primary references used for the current design:

- Binance Spot testnet and market data docs: https://developers.binance.com/docs/binance-spot-api-docs/testnet and https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- Binance WebSocket stream constraints: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
- Binance USD-M futures leverage endpoints: https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Notional-and-Leverage-Brackets and https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Change-Initial-Leverage
- NautilusTrader backtesting concepts: https://nautilustrader.io/docs/latest/concepts/backtesting/
- QuantConnect slippage modeling concepts: https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts

## Implemented Assumptions

Execution cost is symbol-specific where market data exists:

- `ticker/24hr` supplies quote volume and trade count.
- `ticker/bookTicker` supplies bid/ask spread.
- `exchangeInfo` proves the symbol exists and is trading.
- Strategy thresholds decide if quote volume, trade count, spread, and liquidity score are acceptable.
- Automatic universe ranking may derive volume and trade-count floors from the
  current quote-asset leaders when static defaults exceed the available market,
  but it keeps hard minimum floors and never relaxes spread, structural
  leveraged-token, or likely pegged-pair filters.

Backtest fill price uses:

- half-spread cost,
- configured slippage,
- latency buffer,
- market-impact cost based on order participation versus candle-volume notional,
- testnet liquidity haircut,
- volatility buffer,
- taker fees.

Every backtest now keeps path evidence, not only a final P&L scalar:

- mark-to-market equity points with drawdown and position side,
- net trade P&L after entry and exit fees,
- trade returns relative to account equity at entry,
- a compact trade log with open/close timestamps, side, notional, fees, and
  return.
- path-quality metrics: gross profit/loss, finite profit factor, expectancy,
  average trade return, trade-return dispersion, and max consecutive losses.

`backtest-chart` renders this actual equity path instead of a synthetic
start/mid/end curve, and model-lab robustness gates can use trade-return
samples when enough trades exist.

Position sizing is stop-loss-budget based. `risk_per_trade` is interpreted as
the equity budget that may be lost if the configured stop-loss is hit; gross
notional is then capped by max position size, leverage, exchange constraints,
and available cash. This same notional calculation is used by risk reporting,
backtesting, live/testnet order sizing, and the buy-and-hold edge baseline.

Model-lab acceptance adds stress and temporal robustness gates before a symbol
is marked accepted. Each saved objective model is replayed with the selected
symbol's measured spread/liquidity profile and must remain profitable under:

- baseline measured execution assumptions,
- wider spread and slippage,
- latency spike with a liquidity haircut,
- combined liquidity crunch, higher fee, wider spread, and latency stress.

If any required scenario fails the objective gates, `model-lab` writes
`stress_validation.json` for that symbol and rejects the candidate. This is
intentional fail-closed behavior; a single profitable replay is not enough.

After stress validation, the exact serialized final model is also replayed over
separate chronological windows. `temporal_robustness.json` records accepted
window count, latest-window status, worst P&L, worst drawdown, and deterministic
market-regime evidence. Conservative objectives require the highest window pass
rate, regular objectives use the middle policy, and aggressive objectives allow
more dispersion while still requiring multiple profitable windows.

The temporal report also includes a statistical edge gate. It computes an exact
sign-test p-value for positive windows and a deterministic bootstrap lower
confidence bound for mean window return. A symbol is rejected if the final model
looks like a lucky aggregate winner rather than a repeatable window-level edge.
The same report summarizes realized P&L, accepted windows, profit factor, and
expectancy by detected regime so operators can see when an apparent edge is
concentrated in one market state.

The training suite also gates selected candidates with purged chronological
walk-forward folds when enough rows are available. The purge gap is at least the
model label lookahead, so rows whose labels can see into a test fold are not
used as training examples for that fold.

Futures safety:

- Binance can support larger initial leverage values, but this app hard-caps autonomous leverage at `20x`.
- Default leverage is `1x`.
- Shorting is only available on futures mode.
- Liquidation buffer is part of strategy config and risk reporting.

Authenticated order reconciliation:

- Paper-mode fills may use simulated fallback quantities because no exchange order exists.
- Live/testnet signed orders must prove execution from exchange fields such as `fills`, `executedQty`, cumulative quote quantity, or an authenticated order-status query.
- A placement ACK that contains `origQty` but no executed quantity is treated as unresolved, not filled.
- If an order response has an `orderId` or client order id but no fill, the app queries the spot `/api/v3/order` or futures `/fapi/v1/order` status endpoint before updating the local ledger.
- If the fill is still unresolved, the loop records an order error and stops instead of silently assuming exposure changed.
- `simple-ai-trading reconcile` compares signed account exposure with the local autonomous ledger. Paper positions are reported but ignored for exchange mismatch math; non-paper local positions must match spot balances or futures positions for the active symbol set.
- Reconciliation writes `data/autonomous/reconciliation.json` and exits nonzero on exchange-only exposure, local-only exposure, or quantity mismatches.

## Testnet vs Mainnet

Testnet fills, liquidity, queue position, and response times can diverge from live markets. The simulation therefore does not treat testnet as a perfect proxy. It applies conservative liquidity haircuts and latency buffers, and it requires per-symbol liquidity evidence before a symbol can join the trading universe.

Known limitations:

- Direct order-book depth and queue position are not yet replayed tick-by-tick.
- Free VRAM is not exposed reliably by DirectML; the app verifies GPU backend functionality and reports unknown VRAM as a warning.
- External news/sentiment sources are still broad crypto-oriented; the liquidity gate is the primary automatic asset filter.
- The current stress model uses top-of-book and candle-volume proxies. It is
  stricter than flat slippage, but still weaker than full L2 order-book replay.
- Very small datasets are marked as insufficient for purged walk-forward gates;
  they are useful for unit tests and smoke checks, not production acceptance.
- Temporal robustness currently uses candle-window regime evidence; full
  order-book regime replay remains a future improvement after depth snapshots
  are stored.
- Statistical edge prefers trade-return samples when enough trades exist and
  falls back to window-level P&L for sparse strategies. It is stricter than no
  statistical screen, but still weaker than a full intrabar order-book
  deflated-Sharpe implementation.

## Operator Rule

Do not interpret a profitable backtest as approval to trade real money. A candidate must pass:

- `compute`
- `ai` if AI is enabled
- `universe`
- `risk`
- `reconcile`
- `audit`
- `backtest`
- `backtest-chart`
- `model-lab` stress, temporal robustness, regime evidence, statistical edge,
  and portfolio gates
- paper or testnet run review

The project remains non-mainnet-first.
