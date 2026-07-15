# Agent Start

This is the smallest safe entry point for work in this repository. It routes
agents to canonical evidence without replacing that evidence.

## Non-negotiable truth

- Scope is BTC, ETH, and SOL. Binance is testnet/Demo or paper only. Polymarket
  is paper and research only. No mainnet or live-money authority exists.
- Conservative is the default profile. Leverage is a risk ceiling, never a
  source of edge. Profitability, ROI, and drawdown claims require reproducible
  source-bound after-cost evidence.
- Risk, reconciliation, Stop, and ownership checks are deterministic. AI may
  veto or downsize only after matched uplift evidence and may never block a
  close or override a safety gate.
- Historical labels, future books, resolutions, fills, and PnL must never enter
  a live inference payload. Unknown order state blocks new exposure.
- Secrets must never enter prompts, logs, artifacts, tests, commits, or docs.

## Task routing

| Task | Read first | Canonical evidence |
|---|---|---|
| Binance model or backtest | nearest model module and test | `docs/model-research/action-value/latest/README.md`, then selected rows from `progress.csv` |
| Prior model failure | last row plus the relevant mechanism row in `docs/model-research/action-value/latest/progress.csv` | that row's named design/report only |
| Polymarket model | `src/simple_ai_trading/polymarket_model.py` and `tests/test_polymarket_model.py` | `docs/model-research/polymarket/latest/README.md` and the nearest numbered contract |
| Polymarket recorder/replay | matching recorder or replay module and test | `docs/model-research/polymarket/prospective-continuity-contract-v2.json` |
| Risk or execution | nearest risk/execution module and test | `docs/LIVE_MARKET_SIMULATION.md` or `docs/POLYMARKET_PAPER_TRADING.md` only at the relevant heading |
| AI provider/model | nearest AI module and test | `docs/AI_MODEL_SELECTION.md` and `docs/ai_model_benchmark_latest.json` |
| CLI | command handler, parser definition, and CLI tests | parser-generated help; do not infer parity from docs |
| Windows app | `src/simple_ai_trading/windows_app.py` and its UI/parity tests | `native/windows/generated/command_contract.hpp` and `tests/test_ai_runtime_and_parity.py` |
| CI/release | one workflow and its test/lint config | `docs/AGENT_WORKFLOWS.md` |
| Broad architecture | `docs/SIMILAR_TRADING_REPOS_REVIEW.md` | source and tests for each affected boundary |

## Model research state

- The compact cross-round ledger is
  `docs/model-research/action-value/latest/progress.csv`. Read its header and
  only the last few rows unless a task names an older mechanism.
- Round 61 rejected elevated-funding spot/perpetual carry on capacity, median
  after-cost return, and lower-confidence-bound gates. Do not tune or retrain
  that family.
- Polymarket currently has a market-anchored baseline, purged BTC/ETH/SOL
  splits, exact depth/fee/latency replay, causal resolution-time cash locking,
  hash-bound label-free inference, a frozen causal action-value contract, an
  implemented hash-persistent ridge baseline, and a preregistered warning-free
  nonlinear challenger. No Round 9 model has been fitted or scored.
  Post-contract continuity-qualified outcomes and prospective results are still
  pending, so no profitability or execution authority exists.
- Round 9 MLP report v2 requires positive validation stress-utility uplift over
  ridge and at least 30 untouched synchronized test groups before reading its
  test partition. Do not weaken or bypass either admission gate.
- Generic finance-LLM benchmark scores are not alpha evidence. Kronos failed
  the repository's causal random-walk benchmark. Any future AI treatment must
  beat the same-period non-AI path after costs without worsening tail risk.

## Efficient workflow

1. Inspect `git status` and the nearest source/test pair.
2. Use exact `rg` queries. Use CocoIndex only for genuinely broad semantic
   routing; confirm its candidates in source.
3. Freeze causal inputs, costs, roles, and rejection gates before reading a new
   evaluation outcome.
4. Run the smallest focused regression during development. Run the complete
   affected-domain suite once at the behavior checkpoint; run the repository
   suite only when the change crosses domains or before significant handoff.
5. Keep numeric evidence in canonical JSON/CSV and regenerate charts from it.
   Do not duplicate full evidence tables in prose.
6. Record a rejected mechanism in the compact progress ledger so later agents
   do not repeat it.

## Freshness rule

This file is routing context, not result evidence. If it conflicts with a
hash-bound report, source code, or test, the canonical artifact wins and this
file must be corrected in the same change.
