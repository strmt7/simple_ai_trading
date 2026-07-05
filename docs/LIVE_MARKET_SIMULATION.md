# Live-Market Simulation Notes

Simple AI Trading backtests are intentionally pessimistic. A strategy that only works in a frictionless candle replay is not acceptable for autonomous day trading.

## Research Baseline

Primary references used for the current design:

- Binance Spot testnet and market data docs: https://developers.binance.com/docs/binance-spot-api-docs/testnet and https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- Binance USD-M futures kline docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data
- Binance rate-limit guidance for backing off on `429`/`418` and tracking request weight: https://developers.binance.com/docs/binance-spot-api-docs/websocket-api/rate-limits
- Binance WebSocket stream constraints: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
- Binance USD-M futures leverage endpoints: https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Notional-and-Leverage-Brackets and https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Change-Initial-Leverage
- NautilusTrader backtesting concepts: https://nautilustrader.io/docs/latest/concepts/backtesting/
- QuantConnect slippage modeling concepts: https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts

## Implemented Controls and Evidence

Execution cost is symbol-specific where market data exists:

- `ticker/24hr` supplies quote volume and trade count.
- `ticker/bookTicker` supplies bid/ask spread.
- `data-sync` now persists a typed top-of-book history with bid/ask price,
  bid/ask quantity, mid price, spread bps, and top-level notional depth in
  SQLite, while still retaining the raw exchange payload for audit.
- `archive-sync` ingests official Binance public archive ZIPs directly into
  the same SQLite store. This is the preferred path for 1-second spot history
  because Binance REST klines support `1s` but paging years of second-bars
  through 1,000-row REST pages is unnecessarily expensive. USD-M futures `1s`
  evidence uses official Binance `aggTrades` archives aggregated into
  one-second candles because USD-M futures kline archives do not publish `1s`.
  No-trade seconds are carry-forward candles with zero volume and zero trade
  count; they preserve time continuity without inventing traded liquidity.
  Operators can pass `--symbols` for an explicit batch or `--top-symbols N`
  with `--quote-asset` to auto-rank liquid symbols from current exchange
  metadata before archive ingestion. This keeps the BTC/ETH/SOL research path
  automatic without relying on stale liquidity assumptions. Large second-level
  backfills should use `--start-period`, `--end-period`, and `--plan-only`
  first; the plan reports listed, filtered, selected, first, and last archive
  periods plus official S3 byte estimates when available before any ZIP
  download starts. Non-plan archive downloads are blocked above
  `--max-planned-gb` (`50` by default) unless the operator deliberately narrows
  the period window or raises the cap.
- Archive downloads compute SHA-256 locally and, by default, verify Binance's
  `.CHECKSUM` sidecar when it is available. A checksum mismatch fails before
  any candle rows are written. `--require-checksum` makes missing sidecars a
  hard failure for promotion-grade data builds; without it, unavailable
  sidecars are recorded as `checksum_status=unavailable` for audit.
- `data-health` is the pre-training database gate. It emits machine-readable
  row counts, UTC spans, expected rows, coverage ratio, gap count,
  archive-status counts, and checksum-status counts, and it exits nonzero when
  configured minimum rows, maximum gaps, coverage ratio, archive errors,
  checksum mismatches, or required verified checksums are not satisfied.
- Promotion-grade `tools/optimization_round.py` runs can use the same checks
  with `--require-prefilled-data`, `--min-data-rows`,
  `--min-coverage-ratio`, `--max-gap-count`, and
  `--require-verified-checksum`. In that mode optimization records
  `data-health.json` and blocks model training/backtesting for a symbol before
  any network backfill can hide missing or unverified archive data.
- `data-sync --full-history` pages backward through exchange klines with the
  venue maximum request size until no older rows are returned. Recent bounded
  syncs remain available for incremental refreshes, but reports label them as
  recent-limit evidence rather than full available history.
- Sync results record kline request count, rows received, coverage ratio,
  gap count, and Binance used-weight/order-count headers when the exchange
  provides them. This makes API usage cost auditable while keeping paging
  efficient.
- `api-budget` reads that rate-limit state from SQLite and refreshes it at
  most opportunistically. The Windows app bottom bar uses the same command.
  Signed `live --live` startup and authenticated autonomous startup are blocked
  when a current sample shows any known Binance request-weight or order-count
  window is at least 80% consumed, or when the exchange returns a `Retry-After`
  value.
- `backtest`, `backtest-chart`, and `backtest-panel` can consume the latest
  typed top-of-book row with `--execution-db data/market_data.sqlite`. The
  loaded profile is written into run artifacts and panel reports, including
  source, spread bps, top-level depth, snapshot timestamp, stale-data warning,
  and the derived liquidity score.
- `exchangeInfo` proves the symbol exists and is trading.
- Strategy thresholds decide if quote volume, trade count, spread, and liquidity score are acceptable.
- Automatic universe ranking may derive volume and trade-count floors from the
  current quote-asset leaders when static defaults exceed the available market,
  but it keeps hard minimum floors and never relaxes spread, structural
  leveraged-token, or likely pegged-pair filters.
- Day-trading session risk is not a fixed UTC window. Backtests, `live`, and
  the autonomous decision function compare the current bar against trailing
  per-symbol volume and against same UTC weekday/hour/minute-bucket history
  from prior bars. Holidays, partial days, low-liquidity overnight periods,
  and schedule changes are therefore treated as measured low-liquidity
  evidence for that symbol and timestamp, not as an assumption baked into the
  code.

Backtest fill price uses:

- half-spread cost,
- optional symbol-specific spread and top-of-book depth from SQLite,
- configured slippage,
- latency buffer,
- market-impact cost based on order participation versus candle-volume notional,
- testnet liquidity haircut,
- volatility buffer,
- taker fees.

Every backtest now keeps path evidence, not only a final P&L scalar:

- a `data_coverage` record with symbol, market type, interval, UTC date span,
  requested/used history scope, candle counts, row count, gap count, coverage
  ratio, full-history flag, and truth basis,
- mark-to-market equity points with drawdown and position side,
- net trade P&L after entry and exit fees,
- trade returns relative to account equity at entry,
- a compact trade log with open/close timestamps, side, notional, fees, and
  return.
- path-quality metrics: gross profit/loss, finite profit factor, expectancy,
  average trade return, trade-return dispersion, and max consecutive losses.
- pre-entry liquidity-session decisions that show whether a trade was
  down-sized because trailing liquidity or a data-probed same-bucket session
  was below history.

`backtest-chart` renders this actual equity path instead of a three-point
display fallback. When equity timestamps are present, the SVG labels the
simulation start/end dates and duration in days/years. Model-lab robustness
gates can use trade-return samples when enough trades exist.

Position sizing is stop-loss-budget based. `risk_per_trade` is interpreted as
the equity budget that may be lost if the configured stop-loss is hit; gross
notional is then capped by max position size, leverage, exchange constraints,
and available cash. This same notional calculation is used by risk reporting,
backtesting, live/testnet order sizing, and the buy-and-hold edge baseline.

Autonomous live/testnet orders use bot-owned client-order IDs. Live stop/close
paths only submit exchange closes for positions that are still present in the
local ledger, carry this bot ownership proof, and include exchange fill or
acknowledgement evidence. Spot positions require `FILLED` or
`PARTIALLY_FILLED` exchange status before the bot will submit a close, because
spot has no reduce-only protection. Futures positions allow the same filled
statuses and may also close a bot-owned `NEW` or `accepted` order only when an
exchange order ID is present, because the close is sent reduce-only. Pending,
local, canceled, rejected, expired, or hand-edited live ledger entries are
preserved and reported with a concrete rejection reason. Manual or external
exchange exposure is reported by reconciliation and is not touched by the bot.
CLI and shell local-close commands refuse to erase live ledger entries because
that would make exchange exposure stale or untracked.

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
- Risk-profile defaults are `5x` conservative, `10x` regular, and `15x` aggressive in futures mode. Spot mode still resolves to `1x` because spot orders cannot use futures leverage.
- Shorting is only available on futures mode.
- Liquidation buffer is part of strategy config, risk reporting, backtest evidence, and promotion gates.
- Futures backtests use a conservative isolated-margin liquidation proxy derived from the same maintenance-margin concept used by Binance futures: `margin_balance = isolated_margin + unrealized_pnl`; if that balance is less than or equal to `current_notional * liquidation_buffer_pct`, the isolated margin is treated as lost, the position is cleared, `liquidation_events` is incremented, and no threshold, objective, market-edge, stress, temporal, or optimization report can be accepted.
- The proxy deliberately fails closed when historical rows gap through a liquidation boundary. It does not assume a favorable stop-loss fill after the maintenance-plus-buffer condition has already been breached.
- Leverage is subordinate to stop-loss-sized position sizing, position caps, daily/session loss budgets, exchange brackets, and reconciliation gates. A higher default does not authorize larger loss-at-stop budgets.

Authenticated order reconciliation:

- Paper-mode fills may use simulated fallback quantities because no exchange order exists.
- Live/testnet signed orders must prove execution from exchange fields such as `fills`, `executedQty`, cumulative quote quantity, or an authenticated order-status query.
- A placement ACK that contains `origQty` but no executed quantity is treated as unresolved, not filled.
- If an order response has an `orderId` or client order id but no fill, the app queries the spot `/api/v3/order` or futures `/fapi/v1/order` status endpoint before updating the local ledger.
- If the fill is still unresolved, the loop records an order error and stops instead of silently assuming exposure changed.
- `simple-ai-trading reconcile` compares signed account exposure with the local autonomous ledger. Paper positions are reported but ignored for exchange mismatch math; non-paper local positions must match spot balances or futures positions for the active symbol set.
- Reconciliation writes `data/autonomous/reconciliation.json` and exits nonzero on exchange-only exposure, local-only exposure, or quantity mismatches.
- Signed `live --live` startup runs the same reconciliation check before it is allowed
  to resume exposure. A pre-existing exchange position without a matching
  bot-owned ledger position is a startup block, not an implicit adoption. This
  keeps the CLI from closing or modifying positions that were not opened and
  tracked by the bot.
- Signed `live --live` entries stamp exchange orders with `sait-o-*` client
  order ids and immediately write the resolved fill into the local bot ledger.
  Signed live closes stamp `sait-c-*` client order ids, write a closed-trade
  ledger entry, and remove or shrink the open ledger position after the resolved
  fill. If a submit call fails after reaching the exchange, the order wrapper
  queries the deterministic client order id before declaring an order error.

Autonomous network-interruption recovery:

- The `live` CLI loop treats Binance market-data failures as retryable recovery
  state. It records `market_error_retry` events, keeps polling at the configured
  cadence, and if the finite run ends before a clean market read it exits
  nonzero with `market_recovery_pending`.
- The first successful `live` market read after an interruption is an observation
  step, not entry permission. Fresh entries are blocked through the shared entry
  risk gate with `recovery_pending=true`, a `recovery_observation` event is
  persisted, and the loop waits through `recovery_cooldown_seconds` when
  configured. Existing open positions still pass through the normal stop,
  take-profit, and emergency drawdown logic on the recovered quote.
- A Binance/network exception does not trigger a new trade, and it does not
  assume the account is flat. The autonomous loop records a heartbeat that says
  `reconcile-before-resume` and keeps retrying at the configured cadence.
- After connectivity returns, the first successful market read is treated as a
  recovery transition. If the run is authenticated, signed account exposure is
  reconciled against the local autonomous ledger before any entry logic runs.
- If reconciliation finds exchange-only exposure, local-only exposure, or a
  quantity mismatch, the loop exits fail-closed. It does not close exposure
  that is not represented in the bot ledger.
- If reconciliation is clean, the loop checks hard daily loss, session loss,
  and consecutive-loss budgets. Breached daily/session budgets close locally
  tracked bot positions at the latest mark and stop the loop; loss-streak
  lockout stops new entries.
- A clean recovery still writes a recovery-observation heartbeat and skips that
  iteration before normal entry logic can resume. It waits through
  `recovery_cooldown_seconds` when configured; a zero-second cooldown still
  gets one no-entry observation cycle.

## Testnet vs Mainnet

Testnet fills, liquidity, queue position, and response times can diverge from live markets. The simulation therefore does not treat testnet as a perfect proxy. It applies conservative liquidity haircuts and latency buffers, and it requires per-symbol liquidity evidence before a symbol can join the trading universe.

Live monitoring should use exchange streams for open/trading symbols whenever
possible. REST polling remains useful for bounded preflight and recovery checks,
but second/subsecond live observation must not burn through REST request weight
when Binance WebSocket streams can carry trade/book/kline updates. The local
supervisor must treat stale market data as a new-entry block, not as permission
to reuse the last signal.

Market modes are handled as exposure controls, not as a reason to force trades.
When regime evidence, model confidence, entropy, loss streaks, or post-outage
recovery indicate that the current market is not predictable enough, the
correct action is to wait. That wait can last for many days if the configured
cooldowns and repeated checks keep failing; the UI should show a simple
waiting/blocked state while the detailed evidence remains in reports.

The live and autonomous loops now apply this as an explicit entry-risk gate.
On every step, they classify the rolling point-in-time model rows, compute a
deterministic `regime_unpredictability_score`, and compare that score with
`StrategyConfig.max_regime_unpredictability`. The live loop also records
`regime_unpredictability_gate` events with regime name, confidence, score,
threshold, notes, and cooldown remaining. Conservative defaults block at a
lower score than regular or aggressive profiles. An active cooldown blocks new
live entries before order sizing, while existing bot-owned positions still
follow their normal stop/take/close logic.

Signed live startup also checks the final model artifact. `model-lab` must stamp
`execution_validation` into the serialized model after the symbol passes
liquidity selection, stress replay, temporal robustness, and final portfolio
risk. This keeps a generic candle-trained model, or an individually strong
symbol from a rejected portfolio, from being treated as live-ready just because
it deserializes and has a positive selection score.

Data provenance is a safety gate, not an annotation. Model-lab rejects accepted
symbols when coverage evidence is missing, has no model rows, has detected
gaps, or falls below the coverage threshold. It also stamps data coverage into
the model's `execution_validation` block so a promoted model carries the exact
timescale and truth basis used for its promotion.

Known limitations:

- Full L2 order-book depth and queue position are not yet replayed tick-by-tick.
- Data-probed session liquidity is only as good as the available exchange
  history for that symbol and interval. A newly listed symbol or sparse archive
  cannot prove historical session behavior and should fail promotion-grade data
  coverage gates rather than be treated as known-liquid.
- The default database is SQLite for zero-service local installation. It uses
  compact numeric columns and WAL mode, but standard SQLite is not a compressed
  time-series engine. Very large 1-second or tick archives should eventually be
  migrated to PostgreSQL/TimescaleDB or DuckDB when full built-in compression
  and columnar/time-series storage become operational requirements.
- Free VRAM is not exposed reliably by DirectML; the app verifies GPU backend functionality and reports unknown VRAM as a warning.
- External news/sentiment sources are still broad crypto-oriented; the liquidity gate is the primary automatic asset filter.
- The current stress model uses top-of-book and candle-volume proxies. It is
  stricter than flat slippage, but still weaker than full L2 order-book replay.
- Very small datasets are marked as insufficient for purged walk-forward gates;
  they are useful for unit tests and smoke checks, not production acceptance.
- Recent-limit kline pulls are useful for smoke checks, but they are not
  equivalent to full-history evidence and are labeled accordingly.
- Temporal robustness currently uses candle-window regime evidence; full
  order-book regime replay remains a future improvement after L2 depth
  snapshots are stored.
- Statistical edge prefers trade-return samples when enough trades exist and
  falls back to window-level P&L for sparse strategies. It is stricter than no
  statistical screen, but still weaker than a full intrabar order-book
  deflated-Sharpe implementation.

## Operator Rule

Do not interpret a profitable backtest as approval to trade real money. A candidate must pass:

- `compute`
- `ai` if AI is enabled
- accepted AI-vs-ML uplift evidence when AI-assisted signal features are enabled
- `universe`
- `risk`
- `coordinator`
- `reconcile`
- `audit`
- `backtest`
- `backtest-chart`
- `model-lab` stress, temporal robustness, regime evidence, statistical edge,
  and portfolio gates
- paper or testnet run review

The project remains non-mainnet-first.
