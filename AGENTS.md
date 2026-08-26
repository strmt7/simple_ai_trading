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
- A public fill after an oracle deadline does not prove its maker order was
  created after finality. Any finalized-winner latency study must bind the exact
  undisputed non-ignore on-chain state before hypothetical or owned order entry;
  a bid resting before that state is directional exposure, not structural carry.
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
- Edge discovery is the current priority. Public, unauthenticated, read-only
  source, market-data, and blockchain research may proceed iteratively when
  each request tests a distinct question or materially refreshes stale evidence.
  Record request bounds and retain raw evidence, but do not require a commit,
  push, or hosted CI run before exploratory public requests. The stricter
  frozen one-use workflow remains mandatory for authenticated, account-specific,
  funded, order-capable, or state-changing operations.
- Before every venue HTTP request, classify the exact endpoint and method from
  a current primary API contract. Never infer that an endpoint is public from
  its path, product name, nearby public endpoints, or an unauthenticated error;
  if the security classification is absent or contradictory, do not call it.
- HTTP `GET` does not prove read-only semantics. Classify account mutation from
  the documented operation as well as the verb; Binance
  `/sapi/v1/soft-staking/set` changes activation state and requires separate
  mutation authority. Never credit Soft Staking yield to pending-order frozen
  assets, Auto-Subscribe allocations, or inventory needed for prompt liquidity.
- Conflicting current primary-source terms are an unresolved gate. Do not pick
  a rate, fee, eligibility rule, or effective date by page location, apparent
  recency, or convenience; preserve both sources and require an explicit
  effective-date source or realized post-change evidence before promotion.
- A displayed `Max` or `Up to` APR is not a full-principal account rate. Bind
  the exact base and bonus tiers, caps, minimum, end time, account eligibility,
  and fees before promotion. A public calculator is only a sensitivity: if its
  horizon, tier, or campaign state is not source-exposed, do not reverse-engineer
  those terms from outputs or call the estimate realized economics.
- Before sampling variants of a reward endpoint, source-bind its filter and
  aggregation semantics and request the documented superset once. For
  Polymarket raw market rewards, `sponsored=true` folds sponsored daily rates
  into `rate_per_day`; a default response cannot close total reward funding,
  and Gamma spread/size metadata does not prove a current funded pool.
- Cross-token cost and reward comparisons must use a source-bound executable
  conversion into one exact unit. A one-for-one stablecoin assumption or a
  different quote currency is a labeled sensitivity only and cannot support an
  after-cost profitability claim. Public evidence that an automation contract
  captured a reward also does not prove independent access to its batching,
  relayer, permissions, fees, or first-block execution path.
- Treat venue fee promotions as exact pair-and-order-role overlays, never as a
  reason to change quote inventory or create volume. A zero maker fee may be
  credited only to an owned maker fill on the currently listed pair; it cannot
  be double-counted with a BNB discount, rebate, or reward. Any quote-asset
  switch first requires its own executable spread, basis, conversion, fill,
  settlement, and opportunity-cost proof.
- Before freezing a time-bounded prospective capture, prove that its duration,
  phase alignment, retained observation tail, and required source timestamps can
  supply every minimum-sample gate. Elapsed duration and zero transport gaps do
  not prove analyzable intervals; record source-boundary continuity separately.
  If an unchanged mechanism already fails an economic gate, do not spend another
  capture merely to repair sample count unless a precommitted decision could
  still change.
- Before claiming complete coverage from a paginated public catalog, prove a
  source-bound population/page ceiling or freeze an explicitly partial rank or
  cursor boundary. A non-null cursor at the request ceiling is incomplete, not
  a zero-candidate universe, and does not justify a larger adaptive rerun.
- For weighted or rate-limited APIs, freeze both the cumulative request weight
  and a limit-derived pacing schedule. After the first retained page, validate
  ordering, timestamp phase, cursor progression, and cross-source alignment on
  retained evidence before continuing pagination; successful JSON parsing alone
  is not an aggregation preflight.

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
   Before a multi-request public screen, validate parsing and aggregation on one
   retained response. Use non-alias helper names and property-bearing objects;
   retain raw responses before downstream calculations so a local calculation
   failure reuses evidence instead of refetching it. For a large discovery or
   inventory response, persist it and print only a bounded aggregate in the same
   request; never stream the full payload to the console as the only copy.
5. Keep edits scoped, match existing patterns, remove only resulting orphans,
   and never revert unrelated work.
6. Keep numeric evidence in canonical JSON/CSV and regenerate charts from it.
   Generated charts and prose are not result authority.
7. Format code before generating implementation-hash-bound evidence; later
   source edits require regeneration.

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
