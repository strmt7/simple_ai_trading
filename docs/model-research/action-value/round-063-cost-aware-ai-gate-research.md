# Round 63: Cost-aware labels and AI evidence binding

**Status:** implementation and contract validation only. No new economic backtest was run, so this round makes no profitability, ROI, drawdown, AI-uplift, testnet, or live-trading claim.

## Defects removed

1. Candidate tuning rebuilt `StrategyConfig` from a partial field list. New safeguards added elsewhere could therefore disappear during model search. Candidate overlays now use `dataclasses.replace`, preserving every non-tuned risk control.
2. Short-horizon labels could target gross moves below configured round-trip taker fees and spread. Training now enforces:

   `gross label barrier > 2 x taker fee + configured spread floor`

   Latency, impact, and testnet-to-live buffers remain additional execution costs in backtests and stress validation. They are not silently treated as expected label costs.
3. AI review used the current UI strategy label threshold rather than the trained artifact's exact feature signature. It now extracts the frozen label mode, horizon, and barrier from the model artifact.
4. A schema-valid LLM approval could rely on incomplete aggregate evidence. Approval now additionally requires passed calibration, multiple-trial control, sealed terminal holdout, accepted after-cost market-edge statistics, financial sanity, no liquidation, walk-forward, stress, temporal, portfolio, and futures microstructure evidence. The model barrier must exceed the exact configured fee-plus-spread floor.
5. The provider could consume tokens when those deterministic gates made approval impossible or while the same instrument was already open. The coordinator now skips both calls before inference; model-side exit signals continue without waiting for AI.

## Research basis

- [Bysik and Slepaczuk, 2026](https://arxiv.org/abs/2606.00060) report that gross BTC forecasts can fail after costs and that a transaction-cost hurdle materially changes executable performance under walk-forward evaluation.
- [Zhao et al., 2025](https://arxiv.org/abs/2503.09988) document that transaction costs materially affect high-frequency label construction and class balance.
- [Binance API documentation](https://developers.binance.com/en/docs/products/spot/rest-api) requires production clients to reconcile uncertain order state and exposes rate-limit evidence; exchange observations remain authoritative over local assumptions.

These sources motivate the controls; they do not validate this implementation or establish an edge.

## Verification

- Ruff passed on every changed source and test module.
- 97 AI-assist, uplift, autonomous, and execution-simulation tests passed.
- 77 training-suite and process-parallelism tests passed.
- A regression test proves that a 0.10 label multiplier cannot push the final target back below the cost floor.

The next economic result must come from a fresh, provenance-bound training and untouched confirmation run. Existing result graphs are intentionally unchanged because altering them without new market evidence would be false.
