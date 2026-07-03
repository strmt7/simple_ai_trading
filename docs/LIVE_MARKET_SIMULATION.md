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

Backtest fill price uses:

- half-spread cost,
- configured slippage,
- latency buffer,
- market-impact cost based on order participation,
- testnet liquidity haircut,
- volatility buffer,
- taker fees.

Model-lab acceptance adds an additional stress matrix before a symbol is marked
accepted. Each saved objective model is replayed with the selected symbol's
measured spread/liquidity profile and must remain profitable under:

- baseline measured execution assumptions,
- wider spread and slippage,
- latency spike with a liquidity haircut,
- combined liquidity crunch, higher fee, wider spread, and latency stress.

If any required scenario fails the objective gates, `model-lab` writes
`stress_validation.json` for that symbol and rejects the candidate. This is
intentional fail-closed behavior; a single profitable replay is not enough.

Futures safety:

- Binance can support larger initial leverage values, but this app hard-caps autonomous leverage at `10x`.
- Default leverage is `1x`.
- Shorting is only available on futures mode.
- Liquidation buffer is part of strategy config and risk reporting.

## Testnet vs Mainnet

Testnet fills, liquidity, queue position, and response times can diverge from live markets. The simulation therefore does not treat testnet as a perfect proxy. It applies conservative liquidity haircuts and latency buffers, and it requires per-symbol liquidity evidence before a symbol can join the trading universe.

Known limitations:

- Direct order-book depth and queue position are not yet replayed tick-by-tick.
- Free VRAM is not exposed reliably by DirectML; the app verifies GPU backend functionality and reports unknown VRAM as a warning.
- External news/sentiment sources are still broad crypto-oriented; the liquidity gate is the primary automatic asset filter.
- The current stress model uses top-of-book and candle-volume proxies. It is
  stricter than flat slippage, but still weaker than full L2 order-book replay.

## Operator Rule

Do not interpret a profitable backtest as approval to trade real money. A candidate must pass:

- `compute`
- `ai` if AI is enabled
- `universe`
- `risk`
- `audit`
- `backtest`
- `backtest-chart`
- paper or testnet run review

The project remains non-mainnet-first.
