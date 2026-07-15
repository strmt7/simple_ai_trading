# Polymarket 5-minute paper trading

**Status:** the prospective public-data recorder, fail-closed level-2 replay,
causal feature materializer, shared paper execution contract, manual
evidence-bound open/close actions, and cross-checked official resolution
settlement are implemented. Continuous strategy coordination remains incomplete.
No authenticated order placement, wallet, private key, live-money claim, or
profitability claim is implemented or authorized.

The Polymarket lane targets only BTC, ETH, and SOL 5-minute Up/Down markets.
It reuses the Binance paper-trading lifecycle and risk core. Venue-specific
code may translate market data, binary tokens, fees, fills, and settlement; it
may not fork ownership, reconciliation, outage recovery, or stop semantics.

The lifecycle, risk, and outage sections below are the required parity contract.
The current executable subset is the public recorder, strict replay by default,
explicit segmented reconnect replay, manual aggressive FAK paper open/close,
journal reconciliation, and official-resolution settlement. Stop/Pause
coordination, passive queue replay, empirical latency calibration, automated
strategy/AI decisions, and independent live liveness loops remain incomplete and
must not be represented as available.

```mermaid
flowchart LR
  B["Binance direct stream"] --> R["Timestamped recorder"]
  C["Polymarket RTDS: Binance + Chainlink"] --> R
  O["Polymarket CLOB order book"] --> R
  G["Gamma market metadata"] --> R
  R --> D["Append-only DuckDB evidence"]
  D --> F["Causal feature dataset"]
  F --> M["Probability and execution models"]
  M --> K["Shared deterministic risk coordinator"]
  K --> P["Shared paper order lifecycle"]
  P --> X["Polymarket fill and settlement adapter"]
```

## Venue truth

- Market discovery comes from Gamma and must prove `recurrence=5m`, active
  order-book trading, exact event start/end times, fee schedule, tick size,
  minimum size, token IDs, and Chainlink resolution source.
- CLOB WebSocket events provide full aggregated books, price-level changes,
  trades, and best bid/ask changes. A reconnect or unprovable gap requires a
  fresh REST snapshot; missing events are never interpolated.
- Polymarket RTDS provides Binance and Chainlink crypto prices for BTC, ETH, and
  SOL. Direct Binance streams are the primary spot-price input; live Chainlink is
  the mandatory settlement-reference input. RTDS Binance is retained as optional
  relay telemetry and is never imputed when absent or stale. Any latency edge must
  be measured prospectively from source timestamps and local monotonic arrival
  clocks; it is never assumed.
- The official market outcome, not a Binance price inference, settles paper
  positions. Finalization requires exact agreement between independently fetched
  closed CLOB and Gamma market records; disagreement remains pending.
- Taker fees are read from each market's current fee schedule and calculated at
  match time. No hard-coded fee curve is allowed.

## Required lifecycle parity

Every paper order uses a deterministic bot-owned intent ID and the shared
idempotency journal. State transitions are append-only:

`INTENT -> SUBMITTED -> ACKNOWLEDGED -> PARTIAL | FILLED | CANCEL_PENDING -> CANCELLED | EXPIRED`

Ambiguous transitions enter `UNKNOWN`, block new exposure, and require
reconciliation. A simulated CLOB match then follows the venue's settlement
shape: `MATCHED -> MINED -> CONFIRMED | RETRYING -> CONFIRMED | FAILED`.
Paper execution remains explicitly simulated; it cannot be presented as an
authenticated `MATCHED`, `MINED`, or `CONFIRMED` user trade.

The future `Stop` action must cancel bot-owned orders and sell only bot-owned outcome inventory by
walking the observed book. If the book cannot absorb the full position, the
remainder stays visibly `CLOSE_PENDING`; the software must not report flat.
Externally opened positions are never adopted, netted, sold, or settled by the
bot. The future `Pause` action must block new intents but continue data, risk, reconciliation,
settlement, and verified close handling.

## Required fill simulation

- Implemented: aggressive FAK paper orders walk exact observed depth after an
  explicit nonzero submission latency and apply recorded fee parameters to each
  fill level.
- Pending: passive orders start behind all displayed quantity at their price. Only
  subsequent opposite aggressive trades at that price consume queue ahead.
  Cancellations receive zero fill credit.
- Implemented: partial fills create inventory only for the filled quantity;
  unfilled FAK quantity is cancelled and a partial close remains visible.
- Pending: submission, market-data, and execution latencies come from prospective
  empirical distributions with a p99 stress replay. Fixed zero latency is
  prohibited.
- Implemented: no synthetic liquidity, midpoint fill, last-price fill, or inferred hidden
  fill is permitted.

## Binary-market risk

The maximum loss at resolution sizes every position. A stop order is a loss
mitigation attempt, not a guaranteed cap, because five-minute books can gap or
empty. The conservative profile is default, profit reinvestment remains off,
and Polymarket leverage is disabled. Hedging means purchasing the opposing
outcome and must include both spreads and fees; naked outcome-token shorting is
not simulated.

The future coordinator must require fresh CLOB, Chainlink, and direct Binance
feeds; synchronized clocks; known fees; sufficient displayed depth; no market
gap; adequate API reserve; and enough time before event close. A strategy that
uses optional RTDS Binance telemetry must separately prove that feed fresh. The
coordinator can abstain for an entire market or day. There is no trade quota.

## Outages and liveness

The future live CLOB heartbeat must cancel resting orders after missed liveness.
Current replay refuses an execution without a later gap-free recorded state and
persists the resulting `UNKNOWN` intent as restart-blocking. Full reconnect
refresh, loss-budget checks, clean-observation recovery, and cooldown handling
remain coordinator work.

The future market-data, model, AI, risk, execution, reconciliation, and
settlement loops must have independent deadlines. That coordinator is not yet
implemented for Polymarket.

## Evidence before model claims

Public price history is minute-fidelity and cannot validate second-level fills
or latency. The first deliverable is therefore a prospective BTC/ETH/SOL CLOB +
RTDS + direct-Binance recorder and paper shadow engine. Strict training admits
only complete gap-free windows with source timestamps, fees, and official
outcomes. An explicit segmented mode can admit validated CLOB reconnect segments
only: every connection change clears reconstructed state, requires fresh token
baselines, and forbids features or simulated latency from crossing the gap. RTDS
or Binance stream gaps are never admitted. AI is a matched optional treatment
and must beat the same ML baseline after spread, fees, depth, latency, partial
fills, and settlement failures.

Each AI treatment retains its exact label-free prompt and raw local-model
response. The publisher reconstructs candidate, permission, decision-delay, and
uplift chains instead of trusting aggregate AI claims.

### Verified prospective run

Research round 2 used one real, gap-free 553.008-second capture from
`2026-07-15T00:46:38.779Z` through `00:55:51.787Z`. The immutable recorder
contains 559,482 raw frames, 559,445 normalized events, 12 market snapshots,
and 12 dual-source official resolutions. Strict replay reconstructed 612,522
book transitions and materialized 16,097 causal candidate states.

The final 46-feature dataset contains 3,458 unique, officially labeled rows
across two in-window resolved markets per asset. Rebuilding it produced the
same dataset hash and an `existing` materialization result. The first market was
already open when capture began and the fourth began after capture ended, so
neither is represented as a complete feature interval.
Bounded database scans then rebuilt the 939.5 MiB evidence store under the
default 1 GiB, two-thread database limits in 73.4 seconds.

| Evidence | Verified value |
| --- | ---: |
| Recorder report SHA-256 | `70dcd66b488dd7c0fb0c22719d7409bc22a48e80465f66b40a7d10577ed06495` |
| Dataset SHA-256 | `a137ffbb32691fdb9be0299f16f339a0e26db43e67379b6532713400c0d2a053` |
| BTC / ETH / SOL feature rows | 1,278 / 1,158 / 1,022 |
| Null labels / temporal violations / stream gaps | 0 / 0 / 0 |

This run validates the recorder-to-label pipeline only. The diagnostic command
used a one-market minimum; production model fitting remains blocked by the
default requirement of at least 30 featured resolved markets per asset. No ROI,
accuracy, AI-edge, or profitability graph is valid yet. The exact round report,
market rows, and current coverage chart are in
[`model-research/polymarket`](model-research/polymarket/latest/README.md).

Run the public recorder from either the CLI or the generated Windows command
surface:

```powershell
simple-ai-trading polymarket-record --duration-seconds 660 `
  --database data/polymarket-paper.duckdb
simple-ai-trading polymarket-resolve `
  --database data/polymarket-paper.duckdb
simple-ai-trading polymarket-features `
  --database data/polymarket-paper.duckdb
```

A 300-second run is a connectivity smoke test, not model evidence: depending on
the start phase, it can end before the next market's post-open feature warm-up.
The 660-second example can contain one fully anchored interval in the worst start
phase, but it is still far below the default 30 resolved markets per asset needed
for training. Long-running prospective capture is required for model work.

The recorder writes exact WebSocket frame text, canonical REST evidence,
normalized event indexes, connection gaps, per-market fee/tick/depth metadata,
and the shared append-only paper-order journal into one resource-bounded DuckDB
database. Completion requires BTC/ETH/SOL market evidence plus CLOB, RTDS, and
direct Binance frames. Feature readiness additionally requires live Chainlink,
direct Binance BBO/trades, and executable CLOB state for every asset. RTDS
history and live updates are counted separately. A run with a reconnect gap is
`degraded`; malformed,
incomplete, hash-inconsistent, post-finalization, or report-count-mismatched
evidence is rejected. Binance spot
`bookTicker` frames currently do not carry exchange event timestamps, so those
fields remain null instead of receiving an invented time.

Replay rebuilds each outcome token's level-2 book from full `book` snapshots and
`price_change` level replacements/removals, verifies published best bid/ask
checksums and dynamic tick changes, atomically combines split updates sharing a
source transition, and executes only against the first proven state after
nonzero latency. Custom top-of-book events are corroboration rather than the
depth-ordering clock because prospective evidence shows they can precede their
matching depth update. Reusing depth or moving backward in replay time is
blocked. Partial-close dust remains owned until a later executable book or a
hash-bound CLOB/Gamma finalization pays the winning token at `1` and the losing
token at `0`; settlement never masquerades as a CLOB sale.

```powershell
simple-ai-trading polymarket-paper --database data/polymarket-paper.duckdb `
  --action status --json
```

`--allow-segmented-gaps` is an explicit exception on `polymarket-features` and
`polymarket-paper`; omitting it preserves strict gap-free behavior. Reconciliation
revalidates the gap evidence and official-resolution set before every paper
action. Mutation, deletion, an unsupported stream gap, or a missing baseline
blocks operation.

`open`, `close`, and `settle` require explicit immutable event IDs. The command
is generated into the Windows command contract from the same parser, so the CLI
and app cannot acquire separate option sets. `open` and `close` also require an
explicit `--latency-ms`; no unmeasured optimistic default is supplied.

Primary references: [authentication](https://docs.polymarket.com/api-reference/authentication),
[market WebSocket](https://docs.polymarket.com/market-data/websocket/market-channel),
[RTDS](https://docs.polymarket.com/market-data/websocket/rtds),
[official RTDS client](https://github.com/Polymarket/real-time-data-client),
[orders](https://docs.polymarket.com/trading/orders/create),
[order lifecycle](https://docs.polymarket.com/concepts/order-lifecycle),
[fees](https://docs.polymarket.com/trading/fees), and
[rate limits](https://docs.polymarket.com/api-reference/rate-limits).
