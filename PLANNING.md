# Product Direction

Simple AI Trading is a beta BTC/ETH/SOL day-trading research platform. The
near-term objective is trustworthy paper and testnet/Demo operation, not a
profit promise or a shortcut to mainnet.

## Priorities

1. Prove source-complete, causal, after-cost model evidence before promotion.
2. Keep Binance and Polymarket capital, ownership, and execution independent.
3. Preserve one backend contract across the CLI and native Windows app.
4. Make deterministic risk, reconciliation, Pause, and Stop independent of AI.
5. Keep storage, API use, and model experiments bounded and observable.

## Product Defaults

| Setting | Default |
| --- | --- |
| Risk | Conservative |
| AI | Requested, but enabled only after GPU/model/provenance gates pass |
| Reinvest profits | Off |
| Spot leverage | `1x` |
| Futures leverage | `5x` conservative, `10x` regular, `15x` aggressive |
| Maximum futures leverage | `20x` application ceiling |
| Binance execution | Paper or testnet/Demo only |
| Polymarket execution | Disabled; BTC-only live-capable boundary has no authority |

## Promotion Standard

A model advances only with reproducible source provenance, causal splits,
explicit spread/fee/latency/liquidity costs, adequate trade activity, positive
after-cost evidence, bounded drawdown, and untouched holdout results. AI must
also show matched uplift over the frozen machine-learning decision; otherwise
it remains a veto-only research component.

The current state and next admissible experiment are recorded in the
[Binance](docs/model-research/action-value/latest/README.md) and
[Polymarket](docs/model-research/polymarket/latest/README.md) status pages.

## Active Experiment Order

1. Let the fixed Round 27 Stage 1 capture and independent Binance BBO sidecar
   finish without reading feature rows or outcomes early.
2. Run the target-blind campaign, condition, source, and population gates.
3. Materialize the frozen 182-field base, 96-field BBO overlay, and six-field
   Round 29 settlement-state overlay without duplicating raw data.
4. Before target access, finish and hash-bind the Round 29 matched model and
   economic operator. Compare 188 versus 182 fields diagnostically and 284
   versus 278 fields for the primary test.
5. Open outcomes only through the existing one-use gate. Reject any candidate
   that fails matched probability, after-cost, delay, concentration, drawdown,
   or untouched sealed checks.

Round 29 does not reconstruct Chainlink's private TWAP calculation. It only
adds deterministic interactions among already observed exact TWAP margin,
remaining time, variance, and path state. It currently has no result, edge,
profitability claim, or trading authority.

## Resume Here

A new development session must start with
[docs/CONTINUATION.md](docs/CONTINUATION.md). It records the fixed data cutoff,
current authority, completed work, safety invariants, and the next ordered
tasks. Repository evidence and live GitHub state must still be verified before
making changes.
