# Polymarket round 003: market-anchored model contract

**Status:** implemented and tested; prospective profitability is not established.

## Decision model

The model treats the normalized Up/Down CLOB midpoint as the prior and learns only
a bounded correction:

```text
logit(P(Up)) = logit(normalized CLOB midpoint) + clip(beta0 + beta' x, -2, 2)
```

`x` contains causal direct-Binance movement and order flow, live Chainlink basis,
volatility, spread, and paired Polymarket book state. Training-only robust scaling
and winsorization are hash-bound. If no regularized candidate improves validation
log loss by the precommitted margin, the correction is exactly zero.

## Evidence split

- Five fixed decision horizons per market: 240, 180, 120, 60, and 30 seconds.
- Every market has total sample weight one, regardless of update count.
- Contemporaneous BTC, ETH, and SOL markets share one time-group assignment.
- Train, validation, and untouched test groups are chronological and separated by
  purged five-minute groups.
- Both official outcomes and all three assets are required in every split.
- Training fails closed below 30 complete officially resolved markets per asset.

## Economic test

Probability scores alone cannot establish an edge. The paired economic diagnostic:

- chooses at most the first positive after-fee proposal per market;
- uses the venue's per-market taker fee, tick, and minimum order size;
- sets the highest fee-aware limit that retains the required expected edge;
- waits the configured latency, then walks displayed full-depth books with FOK;
- rejects stale, gapped, crossed, future, or insufficient-depth states;
- caps worst-case loss at 0.5% per market and 1.5% per five-minute group;
- settles only from official CLOB/Gamma cross-checked evidence.

Polymarket documents that the midpoint is not executable, buyers pay the ask, FOK
pricing must walk displayed depth, and fees are market-specific at match time. The
implementation follows those contracts rather than crediting midpoint fills
([order book](https://docs.polymarket.com/trading/orderbook),
[prices](https://docs.polymarket.com/concepts/prices-orderbook),
[fees](https://docs.polymarket.com/trading/fees)).

## AI ablation

AI is veto-only. It receives the exact frozen pre-execution ML proposals and causal
features, never labels, settlement, PnL, timestamps, or future fill state. It cannot
create a side, reverse a side, increase size, or relax a price. Invalid, uncertain,
slow, or unavailable output vetoes entry.

Live AMD-host adversarial risk benchmark on 2026-07-15:

| Model | Parameters | Score | Actions | Valid JSON | Mean latency | Gate |
|---|---:|---:|---:|---:|---:|---|
| `qwen3:8b` | 8B | 0.983 | 11/11 | 11/11 | 2.91 s | pass |
| `qwen3.5:9b` | 9B | 0.939 | 10/11 | 11/11 | 3.29 s | reject |
| `fin-r1:8b` | 8B | 0.980 | 11/11 | 11/11 | 17.84 s | reject |

The pass authorizes risk-review research only; it is not market-edge evidence.
AI uplift still requires matched
after-cost ML-vs-AI evidence over at least 90 days; this short prospective capture
cannot satisfy that gate. A newer model is not selected merely because it is newer.

## Rationale

LOB research warns that strong predictive metrics need not become actionable
transactions, so untouched proper scores and full execution are both mandatory
([Briola et al.](https://arxiv.org/abs/2403.09267)). The residual architecture is
deliberately lower capacity than DeepLOB-style sequence models until prospective
sample size supports them. Finance-specific foundation models such as Kronos are
retained as challengers, not assumed improvements
([Kronos](https://arxiv.org/abs/2508.02739)).

No fixture result, benchmark safety score, raw classifier score, or unfilled quote
is a profitability claim or live trading authority.
