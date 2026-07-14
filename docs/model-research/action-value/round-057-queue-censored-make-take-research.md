# Round 57 research: queue-censored make/take action value

**Status:** frozen development rationale. This document makes no profitability,
AI-uplift, leverage, testnet, or live-trading claim.

## Why the model must change

Round 56 improved rank association but still predicted a negative top-quintile
payoff after the frozen 16 bps round-trip stress charge. Tightening its score
threshold after seeing that result would be curve fitting. The failure is more
fundamental: hourly decisions discard the exact order-flow data already stored,
and taker-only entry makes every candidate pay spread and taker fees before its
forecast has earned anything.

Round 57 therefore models the decision the execution engine actually faces:
**make, take, or abstain**. It does not assume that passive execution is free.
A passive fill is valuable only if its fill probability and post-fill return
jointly overcome adverse selection, exit costs, and stop risk.

## Structural support probe

The committed probe reads no strategy return, P&L, or policy threshold. It
uses one checksum-verified UTC day, a $1,000 reference order, 750 ms placement
latency, 10-second decisions, the complete displayed L1 queue ahead, and only
same-price aggressive trade prints. Cancellations receive zero fill credit and
the order itself must be completely printed through.

| Symbol | Bid order filled by 15 s | Ask order filled by 15 s |
|---|---:|---:|
| BTCUSDT | 25.59% | 26.73% |
| ETHUSDT | 25.04% | 24.74% |
| SOLUSDT | 13.43% | 14.46% |

This establishes class support, not economic edge. The canonical external
report is `round57-passive-fill-support-20260714-v2.json`, SHA-256
`d42b7e9d89c417e1a1c83056d98777a3ae311e06dbde137f736f4e51cb92487d`.

## Frozen action contract

At each eligible decision the model scores five mutually exclusive actions:

1. passive long at the observed bid;
2. passive short at the observed ask;
3. aggressive long through the observed ask;
4. aggressive short through the observed bid;
5. abstain.

Passive orders expire after 15 seconds. Full fill requires initial displayed
queue ahead plus the bot order quantity to be consumed by matching aggressive
prints. A cancellation or a quote-level disappearance never creates a
historical fill. This is a conservative lower-bound simulation, not proof of
historical queue position; the archive does not contain order-level L2 events.

For passive actions, the model separately estimates fill-time survival and the
conditional after-cost payoff given fill. It must also estimate conditional
lower-tail payoff and post-fill adverse selection. Expected action value is
fill probability times conditional payoff; an unfilled order has zero P&L and
cannot be silently converted into a market order. Aggressive actions use the
same latency, path, risk, and exit rules but always pay the contemporaneous
spread and taker entry cost.

## Model hierarchy

The first benchmark is intentionally difficult to beat:

- a shared action-conditioned LightGBM model on causal L1, trade-tape,
  aggregate-depth, queue-ahead, and exponentially decayed order-flow features;
- explicit 5/10/15-second fill hazards with proper survival scores;
- conditional mean and 20th-percentile after-cost payoff models;
- one common action scale, chronological calibration, and non-overlapping
  portfolio replay.

A compact DirectML temporal model is benchmarked only after the tree path has
valid targets and positive predictive skill. DeepLOB-style claims are not
appropriate because the archive has L1 plus sampled aggregate depth, not a
message-complete multi-level book. The simple model remains the control because
recent LOB benchmarks show that architecture complexity does not guarantee
actionable transactions.

## AI boundary

The locally verified multibillion-parameter Qwen model may contribute only:

- schema-constrained, value-blind factor programs for queue toxicity and
  regime interactions;
- an asynchronous risk veto that never blocks the execution or reconciliation
  loops;
- a matched AI-vs-ML ablation on identical rows, costs, actions, and replay.

It cannot see labels, returns, symbol identity, or dates while proposing factor
programs. It cannot submit, size, close, or approve an order. Time-MoE 2.4B is
not included merely because it is large: its official runtime is CUDA-oriented,
covariate support remains incomplete, and no DirectML or random-walk uplift has
yet been proved on this host. Kronos remains rejected by the existing causal
benchmark. Parameter count is not accepted as evidence of alpha.

## Two-timescale path after this screen

The existing 30-day exact-BBO corpus is selection-contaminated development
data. A passing mechanism expands to the complete 320-day official BBO
intersection. A separate multi-year one-second trade-derived lane then supplies
slow regime and volatility priors. Exact execution confirmation must remain
prospective because Binance's official `bookTicker` archive does not span
multiple years.

## Research basis

- [Deep attentive survival analysis](https://arxiv.org/abs/2306.05479): model
  passive fill time as a survival distribution and evaluate it with proper
  scores.
- [The Market Maker's Dilemma](https://doi.org/10.2139/ssrn.5074873): a live
  Binance BTC perpetual experiment reports a negative relation between fill
  likelihood and post-fill return, so fill and toxicity must be modeled jointly.
- [Queue-reactive model](https://arxiv.org/abs/1312.0563): order-flow intensity
  depends on queue state and supports queue-aware execution simulation.
- [Deep Limit Order Book Forecasting / LOBFrame](https://arxiv.org/abs/2403.09267):
  forecast metrics need not imply complete executable transactions.
- [Price impact of order-book events](https://arxiv.org/abs/1011.6402):
  order-flow imbalance is depth-dependent rather than a universal raw signal.
- [Binance public archive](https://github.com/binance/binance-public-data):
  checksum-verified official BBO and trade events are the source of record.
- [Binance local-order-book procedure](https://developers.binance.com/legacy-docs/derivatives/coin-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly):
  sequence gaps require a fresh snapshot rather than guessed updates.
- [TLOB](https://arxiv.org/abs/2502.15757): simple, LOB-adapted baselines can
  outperform more complex architectures and predictability changes over time.
