# Round 73: Impact absorption and liquidity recovery

**Status:** feed qualification passed under capture contract v2. Compact v4
storage has passed 30-second and three-minute live probes, but its required
one-hour qualification has not run. No feature comparison, model, replay, or
profit result exists. This round grants no AI, leverage, paper, testnet, or
live authority.

## Why this is different

Round 36 found repeatable five-second directional information in static L1
imbalance, but its best gross move was only `0.4584 bps` and its best delayed
after-cost mean was `-11.5790 bps`. Round 58 rejected value-blind symmetric
touch making. Round 72 rejected all nine BTC/ETH/SOL spot-flow components.

Round 73 therefore does not add another threshold or larger network to those
inputs. It asks a different, event-conditioned question: after an aggressive
flow shock, does the **way the multi-level book is consumed and replenished**
distinguish absorption/reversion from toxic continuation after realistic delay
and costs?

```mermaid
flowchart LR
  A["Verified diff depth + BBO + trades"] --> B["Gap-free local L2 book"]
  B --> C["Aggressive-flow shock"]
  C --> D["Depth consumption"]
  C --> E["Adds / removals / replenishment"]
  C --> F["Impact and recovery"]
  D --> G["Shallow staged model"]
  E --> G
  F --> G
  G -->|"proper-score pass"| H["Frozen quote-path replay"]
  G -->|"fail"| X["Reject"]
  H -->|"after-cost lower bound > 0"| I["Prospective shadow only"]
  H -->|"fail"| X
```

## Data truth

- BTCUSDT, ETHUSDT, and SOLUSDT USD-M perpetuals only.
- Official Binance `depth@100ms`, `bookTicker`, `aggTrade`, `markPrice@1s`,
  `forceOrder`, depth snapshot, exchange metadata, clock, and open interest.
- Every sequence gap, queue overflow, crossed book, product mismatch, or stale
  state invalidates the affected segment. Reconnect means resnapshot and cool
  down; missing events are never filled in.
- The liquidation feed is a throttled snapshot. No message means "not observed",
  not zero liquidations. Public L2 also omits hidden/RPI liquidity and cannot
  identify market makers, whales, spoofing, or manipulation.
- Diff-depth quantity decreases are displayed removals, not observed
  cancellations. Aggregate trades and removals remain separate when their
  attribution is ambiguous; the software never invents an order-lifecycle fact.
- The single evidence store is `data/microstructure.duckdb`. Contract-v2 run
  `5d89804a8f404d9b80b3a3ce2d796561` passed one uninterrupted hour with
  3,988,592 exact-wire messages and an independent full replay audit. It
  authorizes feature-pipeline diagnostics only; the one-hour corpus is far
  below the seven-day viability and thirty-day promotion gates. See
  `round-073-capture-qualification-2026-07-22.json`.
- The v2 indexed row layout was measured before any long capture. Contract v3
  removed redundant per-message strings and primary-key indexes without
  weakening exact-wire replay, but probe `feb1289d71884a23818be1b7f1de3b3e`
  exposed a terminal latency-query defect and is permanently development-only.
  Contract v4 restores provider event time to the compact link and adds an
  absolute DuckDB-plus-WAL cap. Live probes `7ffd4edbd2654b5997704c988802580d`
  and `ec114dd2c28d4641b0158f4bd0b32c72` passed fresh-process replay. They
  authorize only one v4 one-hour qualification attempt, not feature or model
  evaluation. See `round-073-v4-probe-evidence-2026-07-22.json`.

Native crypto spot and perpetual instruments trade continuously and have no
formal daily close. UTC days are statistical blocks only. Bitcoin, ether, or
other exchange-traded products are separate listed instruments: any later ETF
context must use that product's actual venue calendar, including holidays,
early closes, halts, auctions, and verified extended-hours sessions. An ETF
close must never be imputed as a Binance close.

## Gates

The staged comparison is prevalence/zero payoff, linear L1+tape, shallow L1+tape,
L2 state, then L2 impact absorption. Model capacity and rows stay identical.
Impact absorption must beat both L2 state and the frozen L1+tape control on held
out log loss, Brier score, MSE, calibration, dependence-aware uncertainty, and a
one-second stress-delay check.

The first seven complete days are only a bounded viability screen. Promotion
requires at least 30 complete prospective days with the final seven sealed.
Each symbol passes or fails independently; unsupported symbols are disabled.
Portfolio research requires at least two independent symbol passes.

Only the same frozen predictions may enter an unlevered quote-path replay. Entry
and exit walk the synchronized visible book for `$100`, `$1,000`, and `$5,000`
notionals and apply at least `12 bps` of round-trip fees and adverse charge. A
positive point estimate is insufficient: the blocked lower confidence bound of
net expectancy and profit factor must clear zero and one, with at least 100 test
trades and bounded tail risk. The result is still not a fill claim.

## Model and AI boundary

A fixed shallow LightGBM is the primary challenger. A TCN/TLOB-style temporal
model can be separately preregistered only after the shallow feature layer passes
on 30 days for at least two symbols. Reinforcement learning, language-model
forecasting, AI vetoes, and leverage are closed in this round. This prevents
capacity from hiding a failed financial mechanism.

## Primary sources

- [Binance USD-M Futures API](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/Introduction)
- [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402)
- [Multi-Level Order-Flow Imbalance](https://arxiv.org/abs/1907.06230)
- [Order Flows and LOB Resiliency](https://arxiv.org/abs/1708.02715)
- [Deep Limit Order Book Forecasting / LOBFrame](https://arxiv.org/abs/2403.09267)
- [TLOB](https://arxiv.org/abs/2502.15757)
- [State-dependent L2 liquidity transitions](https://arxiv.org/abs/2607.09230)
- [CFTC disruptive-practices guidance](https://www.cftc.gov/LawRegulation/FederalRegister/FinalRules/2013-12365.html)

These sources motivate the experiment. None establishes an edge for this
repository.
