# AGENTS

Read `docs/AGENT_START.md` first. Hash-bound evidence and executable contracts
override prose.

## Hard Rules

- Work in this session only; never use another agent or subagent.
- AI Git history must use `AI agent <>` for author and committer. Commit with
  `git -c user.name='AI agent' -c user.email= commit ...`. Never use a human,
  host, vendor, tool, CI, global-config, or noreply identity. Read
  `docs/AI_COMMIT_IDENTITY.md` before creating or auditing history.
- Binance scope is BTC, ETH, and SOL on testnet/Demo or paper only.
  Polymarket research covers BTC/ETH/SOL; its independent live-capable boundary
  is BTC-only and disabled by default. No live-money authority exists.
- Conservative is default. Leverage is a ceiling, never edge. Profitability,
  readiness, ROI, and drawdown claims require reproducible source-bound,
  after-cost evidence.
- Aggregate performance never establishes an all-regime edge. Apply
  `docs/model-research/cross-regime-edge-acceptance-contract-v1.json`; an
  unsupported bullish, bearish, sideways, choppy, volatility, liquidity, or
  latency slice must reject promotion or abstain from new exposure.
- Risk, ownership, reconciliation, Stop, and close controls are deterministic.
  AI may only veto or reduce risk after matched uplift evidence and may never
  override a safety gate or block a close.
- Polymarket terminal state requires authenticated exact-order evidence or
  exact fill evidence. Stop may cancel and sell only bot-owned hashes and
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

Apply the pinned Karpathy baseline from
`multica-ai/andrej-karpathy-skills@2c606141936f1eeef17fa3043a72095b4765b9c2`:
think first, state material uncertainty, keep changes small, preserve local
contracts, and verify reproducibly. Do not load upstream `EXAMPLES.md`.

1. Inspect `git status`.
2. Read one nearest source file, its matching test, and the relevant local
   skill. Use the canonical artifact routed by `docs/AGENT_START.md` only when
   needed.
3. Use exact `rg` first. For genuinely broad semantic routing, use the external
   `cocoindex-code-search` workflow with at most five results, then confirm each
   candidate in live source. Never build its index during high system load.
4. Freeze causal inputs, costs, roles, rejection gates, and test access before
   viewing a new model outcome.
5. Keep edits scoped. Match clear existing patterns, remove only orphans caused
   by the change, and never revert unrelated work.
6. Keep numeric evidence in canonical JSON/CSV and regenerate charts from it.
   Generated charts and prose are not result authority.

Do not broadly read the README, historical round designs, generated SVG, or
large CSV files. The detailed workflow and imported-tool provenance are in
`docs/AGENT_WORKFLOWS.md`; broad architecture starts with
`docs/SIMILAR_TRADING_REPOS_REVIEW.md`.

## Verification

- During iteration, run the smallest focused test and Ruff check covering the
  changed behavior. Every new branch needs a direct assertion, including the
  normal and fallback sides of error handling.
- At a behavior checkpoint, run the complete affected-domain suite once.
- Run the full pytest and coverage suites only for shared-core changes, release
  preparation, or significant final handoff; do not repeat them after each edit.
- CLI changes require parser/handler coverage and generated native-contract
  parity. Model/backtest changes require contract, causal-split, economic-gate,
  persistence, and tamper tests for that domain.
- Run `tools/update_readme_badges.py --check` after badge changes. The README
  badge block is generated and must not be hand-edited.

Completion requires implemented behavior, focused tests, one relevant live or
artifact validation, synchronized CLI/Windows metadata where applicable, and
truthful documentation of any remaining block.
