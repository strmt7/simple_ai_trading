# Round 64: Positive-expectancy meta-label execution

**Status:** implementation and contract validation only. No new economic backtest was run, so this round makes no profitability, ROI, drawdown, AI-uplift, testnet, or live-trading claim.

## Change

The trained meta-label policy no longer assigns reduced capital to a weaker signal band merely because a stronger band passed. A downsize band is retained only when that exact development bucket has the objective's minimum sample support, positive mean after-cost return, and positive aggregate after-cost P&L. Otherwise its threshold collapses to the take threshold and every weaker signal is skipped.

For each proposed take or downsize action, the shared decision object now carries the exact bucket's sample count, required sample floor, precision, required precision floor, mean after-cost return, and after-cost P&L. Liquidity/session overlays preserve this evidence when they reduce size. AI review fails before inference when the bound bucket lacks its required support or positive after-cost expectancy. Zero-sized deterministic proposals also bypass inference.

The CLI and Windows app consume the same Python decision and AI-assist contracts; no UI-only trading rule was added.

## Research basis

- [Bysik and Slepaczuk, 2026](https://arxiv.org/abs/2606.00060) show why forecast accuracy without explicit transaction-cost treatment is insufficient for executable BTC strategies under walk-forward evaluation.
- [Zhao et al., 2025](https://arxiv.org/abs/2503.09988) describe the material effect of transaction costs on high-frequency label construction and class balance.

These papers motivate cost-aware acceptance. They do not validate this implementation or establish market edge.

## Verification

- 93 focused meta-label, AI-assist, autonomous, and execution tests passed.
- 5 autonomous decision-construction tests passed.
- 2 focused backtest meta-label/liquidity tests passed.
- The sealed-terminal training regression passed.
- Ruff passed on every changed source and test module.

The next graph may change only after a fresh provenance-bound training run and untouched confirmation evaluation. Existing economic evidence remains unchanged.
