# Round 65: Live payoff evidence for AI review

**Status:** implementation and contract validation only. No new economic backtest was run, so this round makes no profitability, ROI, drawdown, AI-uplift, testnet, or live-trading claim.

## Change

The live AI case previously exposed the final direction probability but not the active payoff experts' economic units. A confident probability could therefore be presented without a current expected net-payoff estimate.

`TrainedModel` now derives compact evidence from the same dependency-free inference primitives used by backtest and live scoring. Only positive-weight payoff experts are reported. Evidence is expressed in after-cost basis points and records the proposed side, expert support/opposition/abstention, payoff hurdle, horizon, value-weight coverage, and the expert's development validation support. Calibrated hurdle experts expose long and short expected payoff separately. The evidence report is capped at three experts by default and remains below a 4 KiB regression budget for the representative case.

The AI wrapper recomputes this evidence from the promoted model object and the exact decision feature vector. It rejects any supplied payoff summary that differs from that recomputation. Payoff inference is not duplicated when AI is disabled.

AI approval now requires current-case after-cost support from either:

1. the exact profitable, sufficiently supported meta-label bucket; or
2. an active payoff expert that supports the proposed side, has positive held-out actionable edge, and predicts positive weighted payoff for the proposal.

Aggregate terminal performance alone cannot satisfy the current-edge requirement. Ineligible cases are rejected before local-model inference, so they consume no LLM tokens. This does not change the frozen deterministic trading policy; a fresh provenance-bound training and terminal evaluation is still required.

## Research basis

- [Bysik and Slepaczuk, 2026](https://arxiv.org/abs/2606.00060) report that sign-based BTC forecasts can fail after transaction costs and that a forecast-magnitude cost hurdle changes executable outcomes.
- [FinTradeBench, 2026](https://arxiv.org/abs/2603.19225) reports a persistent gap in LLM numerical and trading-signal reasoning. That supports supplying audited numeric evidence and retaining deterministic gates instead of asking an LLM to infer missing market edge.

These sources motivate the contract; they do not validate this implementation or establish edge.

## Verification

- 149 model, hybrid, AI-assist, and autonomous tests passed together.
- 5 autonomous decision-construction tests passed.
- 2 signed-payoff accelerated batch-scoring tests passed.
- 2 side-specific action-payoff accelerated batch-scoring tests passed.
- 46 focused hybrid-payoff and AI-assist tests passed after the final token-free rejection regression was added.
- A spoofed positive-payoff summary is rejected before provider inference when it differs from the promoted model.
- Ruff and `git diff --check` passed on changed code.

Existing performance graphs remain unchanged because no new market experiment was run.
