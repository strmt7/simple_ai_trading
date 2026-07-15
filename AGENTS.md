# AGENTS

Read `docs/AGENT_START.md` first. It is the compact task router and source of
current trading-research truth. Hash-bound evidence and executable contracts
override prose if they disagree.

## Hard Rules

- Work in this session only. Never spawn, delegate to, or use another agent or
  subagent for exploration, coding, review, or testing.
- AI Git history must use `AI agent <>` for author and committer. Commit with
  `git -c user.name='AI agent' -c user.email= commit ...`. Never use a human,
  host, placeholder, vendor, model, tool, CI, previous-commit, global-config,
  or noreply identity. Read `docs/AI_COMMIT_IDENTITY.md` before creating or
  auditing history.
- Scope is BTC, ETH, and SOL. Binance is testnet/Demo or paper only;
  Polymarket is paper/research only. No live-money authority exists.
- Conservative is the default. Leverage is a ceiling, never evidence of edge.
  Do not claim profitability, production readiness, ROI, or drawdown without
  reproducible source-bound, after-cost evidence.
- Risk, ownership, reconciliation, Stop, and close controls are deterministic.
  AI may only veto or reduce risk after matched uplift evidence and may never
  override a safety gate or block a close.
- Future books, labels, resolutions, fills, and PnL never enter inference.
  Unknown order state blocks new exposure.
- Never print, prompt, log, serialize, test, document, or commit credentials,
  secrets, tokens, signed requests, or unredacted secret fields.
- Preserve testnet, dry-run, diversification, liquidity gating, and the app's
  `20x` leverage cap unless a stricter frozen contract applies.
- No network calls in tests unless explicitly stubbed. Do not hard-code host
  capabilities; detect and record effective backends and fallbacks.

## Working Method

Apply the pinned Karpathy baseline from
`multica-ai/andrej-karpathy-skills@2c606141936f1eeef17fa3043a72095b4765b9c2`:
think before coding, state material uncertainty, prefer the smallest
maintainable change, avoid speculative abstraction, preserve local contracts,
and finish with reproducible verification. Do not load upstream `EXAMPLES.md`
by default.

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
