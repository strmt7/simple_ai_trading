# AGENTS

Read `docs/AGENT_START.md` first. Hash-bound evidence and executable contracts
override prose.

## Hard Rules

- Work in this session only; use no subagent.
- AI Git history must use `AI agent <>` for author and committer. Read
  `docs/AI_COMMIT_IDENTITY.md` before committing or auditing; never use a human,
  host, tool, CI, global-config, or noreply identity.
- Binance is BTC/ETH/SOL testnet, Demo, or paper only. Polymarket research is
  BTC/ETH/SOL; its BTC-only live-capable boundary defaults off. No live-money authority exists.
- Conservative is default. Leverage is a ceiling, never edge. Profitability,
  ROI, readiness, and drawdown claims require source-bound after-cost evidence.
- Aggregate performance never establishes an all-regime edge. Apply
  `docs/model-research/cross-regime-edge-acceptance-contract-v1.json`; an
  unsupported bullish, bearish, sideways, choppy, volatility, liquidity, or
  latency slice must reject promotion or abstain from new exposure.
- Risk, ownership, reconciliation, Stop, and close controls are deterministic.
  AI may only veto or reduce risk after matched uplift evidence and may never
  override a safety gate or block a close.
- Polymarket terminal state requires authenticated exact-order or fill evidence.
  Stop may cancel and sell only bot-owned hashes and
  parent-bound lots; foreign state is never modified.
- Future books, labels, resolutions, fills, and PnL never enter inference.
  Unknown order or redemption state blocks new exposure. Polymarket settlement
  never auto-deploys wallets or creates token approvals.
- Never print, prompt, log, serialize, test, document, or commit credentials,
  secrets, tokens, signed requests, or unredacted secret fields.
- Preserve testnet, dry-run, diversification, liquidity gating, and the app's
  `20x` leverage cap unless a stricter frozen contract applies.
- The installed CLI and native app both invoke `simple_ai_trading.entrypoint`.
  Register independent command extensions there and keep
  `command_contract.py` on the same parser; do not create frontend-only
  Polymarket controls.
- No network calls in tests unless explicitly stubbed. Do not hard-code host
  capabilities; detect and record effective backends and fallbacks.

## Working Method

Use the pinned Karpathy baseline from
`multica-ai/andrej-karpathy-skills@2c606141936f1eeef17fa3043a72095b4765b9c2`:
think first, state material uncertainty, keep changes small, preserve contracts,
and verify reproducibly. Do not load upstream `EXAMPLES.md`.

1. Inspect `git status`.
2. Read the nearest source, matching test, and relevant local skill. Use the
   artifact routed by `docs/AGENT_START.md` only when needed.
3. Use exact `rg` first. For broad semantic routing, use the external
   `cocoindex-code-search` workflow with at most five results, then confirm each
   candidate in live source. Never build its index during high system load.
4. Freeze causal inputs, costs, roles, rejection gates, and test access before
   viewing a new model outcome.
5. Keep edits scoped, match existing patterns, remove only resulting orphans,
   and never revert unrelated work.
6. Keep numeric evidence in canonical JSON/CSV and regenerate charts from it.
   Generated charts and prose are not result authority.

Do not broadly read the README, historical round designs, generated SVG, or
large CSV files. The detailed workflow and imported-tool provenance are in
`docs/AGENT_WORKFLOWS.md`; broad architecture starts with
`docs/SIMILAR_TRADING_REPOS_REVIEW.md`.

## Verification

- During iteration, run the smallest focused test and Ruff check. Every new
  branch needs a direct assertion, including normal and fallback error paths.
- At a behavior checkpoint, run the complete affected-domain suite once.
- Run full pytest and coverage only for shared core, release preparation, or a
  significant final handoff; do not repeat them after each edit.
- CLI changes require parser/handler coverage and generated native-contract
  parity. Model/backtest changes require contract, causal-split, economic-gate,
  persistence, and tamper tests for that domain.
- Run `tools/update_readme_badges.py --check` after badge changes. The README
  badge block is generated and must not be hand-edited.

Completion requires implemented behavior, focused tests, relevant live or
artifact validation, synchronized CLI/Windows metadata, and truthful blockers.
