# Open-Source Gap Analysis And Development Ruleset - 2026-07-05

This document is a working ruleset for improving Simple AI Trading against
current high-star open-source trading and quantitative-finance projects. It is
not performance evidence and must not be used to claim ROI, profitability, or
live readiness.

## Sources Checked

GitHub API metadata and primary project pages were checked on 2026-07-05 for
these representative high-star projects:

- OpenBB-finance/OpenBB: <https://github.com/OpenBB-finance/OpenBB>
- freqtrade/freqtrade: <https://github.com/freqtrade/freqtrade>
- microsoft/qlib: <https://github.com/microsoft/qlib>
- QuantConnect/Lean: <https://github.com/QuantConnect/Lean>
- hummingbot/hummingbot: <https://github.com/hummingbot/hummingbot>
- polakowo/vectorbt: <https://github.com/polakowo/vectorbt>
- jesse-ai/jesse: <https://github.com/jesse-ai/jesse>
- AI4Finance-Foundation/FinRL: <https://github.com/AI4Finance-Foundation/FinRL>
- goldmansachs/gs-quant: <https://github.com/goldmansachs/gs-quant>

The model-risk standard is informed by SR 26-2/OCC 2026 model-risk guidance,
which emphasizes model development/use, validation, outcomes analysis,
monitoring, governance, controls, and critical review of the quality and extent
of evidence. Backtest-overfitting work from Bailey, Borwein, Lopez de Prado,
and Zhu emphasizes that ordinary holdout backtests are unreliable in strategy
selection and that PBO/CSCV-style evidence is needed to estimate false discovery
risk. Recent comparative work on financial ML validation highlights the value
of purged/combinatorial cross-validation under non-stationarity, autocorrelation,
and regime shifts.

Regulatory and research sources:

- Federal Reserve SR 26-2:
  <https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm>
- OCC Bulletin 2026-13:
  <https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html>
- Probability of Backtest Overfitting:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
- Financial Machine Learning Cross Validation: A Comparative Study:
  <https://arxiv.org/abs/2505.15394>

## What Strong Repos Have In Common

- Clear separation between data, research, backtest, execution, and reporting
  layers.
- Documented CLI or API workflows, not only ad hoc notebooks.
- Dry-run/paper modes before live execution.
- Tests, CI, contribution rules, docs, and packaging metadata.
- Exchange or data-provider abstraction layers.
- Persistent storage for state or market data.
- Backtesting and parameter optimization surfaces.
- Visualization or reporting for performance analysis.
- Explicit disclaimers that backtests do not guarantee profit.

## What Is Still Missing Across Open Source

No surveyed project is a complete answer to this repo's target. Common gaps:

- **Backtest-to-live proof gap:** many projects can backtest and live trade, but
  do not force every live-capable model to carry complete data provenance,
  execution assumptions, temporal robustness, stress, portfolio, and
  model-selection evidence.
- **Model-risk governance gap:** most repos do not enforce SR 26-2-style
  conceptual soundness, outcomes analysis, monitoring, and governance controls
  as machine-readable promotion gates.
- **False-discovery gap:** large parameter sweeps are common, but exact
  promotion gates for purged folds, PBO/CSCV-style diagnostics, deflated scores,
  market-edge tests, and bootstrap/sign-test evidence are not uniformly required
  before execution.
- **AI authority gap:** AI/LLM features are often research assistants or signal
  generators. They rarely prove AI-vs-non-AI uplift with paired holdout samples
  and deterministic vetoes before a live-capable decision.
- **Operational safety gap:** exchange connectors exist, but stale-position
  prevention, bot-owned position proof, reduce-only close logic, restart
  reconciliation, network-outage recovery, rate-limit gating, and stop/pause
  semantics are usually not one unified invariant.
- **Microstructure realism gap:** spread/slippage/fees are often configurable,
  but per-symbol liquidity, depth, latency, testnet-to-mainnet gap, low-liquidity
  session detection, and liquidation-buffer effects are not always
  automatically measured and carried into promotion evidence.
- **Parity gap:** GUI and CLI surfaces, when both exist, are often not generated
  from a shared command contract with tests that prevent drift.
- **Research-to-production gap:** notebooks and demos often prove concepts, but
  production-safe promotion needs typed artifacts, audit trails, reproducible
  commands, source hashes, and fail-closed behavior.
- **GPU portability gap:** frameworks may accelerate selected workflows, but
  Windows AMD/NVIDIA/Intel DirectML-style portability and recorded backend
  evidence are uncommon as a hard promotion concern.

## Development Rules For This Repo

1. **Evidence beats claims.** No ROI, P&L, drawdown, win-rate, AI-uplift, or
   optimization claim may be documented unless it comes from real source data
   and carries command, source, symbol, market, interval, UTC span, row counts,
   coverage, execution assumptions, compute backend, and artifact paths.
2. **Promotion gates fail closed.** Missing evidence is failure, not neutral.
   Accepted model-lab outcomes must include real data coverage, financial
   sanity, purged walk-forward evidence, selection-risk/PBO evidence, stress,
   temporal robustness, market edge, portfolio risk, and learning-feedback
   recovery evidence when applicable.
3. **AI never overrides deterministic risk.** AI may advise only after
   deterministic gates pass. AI promotion requires multibillion-model evidence,
   paired AI-vs-ML holdout uplift, sign-test/positive-delta proof, and no tail
   risk degradation.
4. **Backtests must be harder than live.** Simulations should include
   per-symbol spread, depth, fees, slippage, latency, liquidity crunch,
   liquidation proxy, testnet-to-mainnet buffer, and low-liquidity session
   detection. Weak assumptions must make promotion harder, not easier.
5. **No hidden exposure.** Live or testnet execution must prove ownership before
   touching positions. Bot-owned client order IDs, fills/acknowledgements,
   local ledger rows, reconciliation reports, and reduce-only closes are
   required for stale-position prevention.
6. **Autonomy is gated autonomy.** The system may trade only while rate limits,
   network health, reconciliation, loss budgets, regime predictability,
   liquidity, model readiness, and risk policy are all healthy. Stop means
   closing verified bot-owned exposure; pause means no new entries.
7. **Full parity is generated and tested.** CLI, Windows app, and docs must
   share command contracts. New workflows require parity tests before they are
   considered complete.
8. **Use focused tests while developing, full gates before push.** During
   implementation, run the narrow tests that cover the changed behavior. Before
   commit/push, run the full required gate set once on the final state.
9. **Prefer source-backed improvements.** For major model, AI, execution, or UI
   changes, check reputable current sources first. Use primary docs/papers when
   possible. Record the practical conclusion in docs when it affects the design.
10. **Do not fake missing infrastructure.** If a feature is incomplete, label it
    beta/incomplete and block promotion paths that would require it. Never add
    placeholder data, fabricated charts, or unverified benchmark results.

## Immediate Development Priority

The next high-impact gap is promotion-grade walk-forward evidence. A model-lab
outcome must not be accepted merely because the purged walk-forward gate was
skipped for insufficient rows. Accepted outcomes and model execution-validation
stamps must carry per-objective walk-forward evidence with real folds, all folds
accepted, no rejection reason, and stable result payloads.
