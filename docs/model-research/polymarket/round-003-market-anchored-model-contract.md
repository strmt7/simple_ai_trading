# Polymarket round 003: market-anchored model contract

**Status:** implemented and tested; prospective profitability is not established.

## Decision model

The model treats the normalized Up/Down CLOB midpoint as the prior and learns only
a bounded correction:

```text
logit(P(Up)) = logit(normalized CLOB midpoint) + clip(beta0 + beta' x, -2, 2)
```

`x` contains causal direct-Binance movement, realized variation, and trade
imbalance at 100 ms through 5 s, plus live Chainlink basis, spread, and paired
Polymarket book state. It also contains two bounded driftless-diffusion
disagreement proxies:

```text
scale = max(realized_volatility_5s_bps, 1e-6) * sqrt(seconds_left / 5)
z = clip(distance_from_open_bps / scale, -8, 8)
gap = clip(logit(Phi(z)) - logit(normalized CLOB midpoint), -12, 12)
```

One distance uses the direct Binance midpoint and the other removes the live
Binance/Chainlink basis. These are interaction features, not calibrated forecasts;
their coefficients can shrink to zero. Training-only robust scaling and
winsorization are hash-bound. L2 regularization is selected only through three
purged rolling folds inside the training span. The outer validation tail then acts
only as a promotion gate. If the frozen candidate does not improve validation log
loss by the precommitted margin, the correction is exactly zero.

## Evidence split

- Five fixed decision horizons per market: 240, 180, 120, 60, and 30 seconds.
- Every market has total sample weight one, regardless of update count.
- Contemporaneous BTC, ETH, and SOL markets share one time-group assignment.
- Train, validation, and untouched test groups are chronological and separated by
  purged five-minute groups.
- Both official outcomes and all three assets are required in every split.
- Training fails closed below 30 complete officially resolved markets per asset.
- Rows and markets are not treated as independent observations. Confirmatory
  status additionally requires at least 30 untouched shared five-minute test
  groups and a deterministic moving-block 95% interval wholly below zero for
  residual-model minus market-prior log loss.

## Economic test

Probability scores alone cannot establish an edge. The paired economic diagnostic:

- chooses at most the first positive after-fee proposal per market;
- uses the venue's per-market taker fee, tick, and minimum order size;
- sets the highest fee-aware limit that retains the required expected edge;
- reconstructs the latest book received no later than order arrival, never the
  first future update, then walks displayed full depth with FOK;
- evaluates predeclared 50, 100, 250, 500, and 1,000 ms network-latency cases;
- adds each measured local-AI response time before network latency in the AI arm;
- rejects stale, gapped, crossed, future, or insufficient-depth states;
- records every abstaining time group and reserves worst-case risk for any
  indeterminate order state;
- caps worst-case loss at 0.5% per market and 1.5% per five-minute group;
- settles only from official CLOB/Gamma cross-checked evidence.

Polymarket documents that the midpoint is not executable, buyers pay the ask, FOK
pricing must walk displayed depth, and fees are market-specific at match time. The
implementation follows those contracts rather than crediting midpoint fills
([order book](https://docs.polymarket.com/trading/orderbook),
[prices](https://docs.polymarket.com/concepts/prices-orderbook),
[fees](https://docs.polymarket.com/trading/fees)).

The July 2026 contract is CLOB V2. FOK orders must fill entirely or cancel, a BUY
market-order amount is denominated in quote currency, its price is a worst-price
limit, and `tick_size_change` events are critical inputs. Public market data uses
WebSocket updates rather than polling; trading endpoints have separate burst and
sustained limits
([create order](https://docs.polymarket.com/trading/orders/create),
[market channel](https://docs.polymarket.com/market-data/websocket/market-channel),
[rate limits](https://docs.polymarket.com/api-reference/rate-limits),
[changelog](https://docs.polymarket.com/changelog)). The current implementation
is paper research only. A future authenticated adapter must translate the
share-based internal intent into the V2 BUY quote amount and reconcile the user
channel before it can receive any live authority.

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

Recent returns, transaction imbalance, book imbalance, and timeliness are retained
because broad high-frequency evidence identifies them as recurring short-horizon
predictors, while also showing that their value decays rapidly with latency
([Ait-Sahalia et al.](https://www.nber.org/papers/w30366)). Polymarket trade-side
inference is deliberately excluded: current venue-specific evidence finds that
public-feed inference often disagrees with authoritative on-chain fills
([Dubach](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6658364)). It may be
added only after causal `OrderFilled` capture and independent reconciliation exist.

No fixture result, benchmark safety score, raw classifier score, or unfilled quote
is a profitability claim or live trading authority.
