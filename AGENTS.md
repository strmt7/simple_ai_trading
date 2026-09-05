# AGENTS

Read `docs/AGENT_START.md` first. Hash-bound evidence and executable contracts
override prose.

## September 4 review session interpretation

For the comprehensive review session explicitly resumed on September 4, use
`docs/REVIEW_2026_09_04.md` to distinguish guaranteed-return proof, expected-value
research, stress scenarios, and scoped snapshot results. This is a session-only
clarification of research routing, not a permanent relaxation of trading or
capture rules. A missing guaranteed floor is not proof of negative expected
value; a failed snapshot is not a family-wide impossibility proof. Retained
public bytes may support explicitly exploratory offline analysis, never a new
independent validation claim. Consumed contracts, exact retry triggers,
protected captures, credential restrictions, and account/order authority remain
unchanged. Do not reuse the legacy logical-parity runner as proof of settlement
identity; its deadline grouping does not bind observation starts (review R4).

September 5 session-only exception, under the user's explicit permission to
revise research rules for this session: rank 16 may run exactly the separately
frozen `docs/review/2026-09-05/triangle-window/contract.json` public quote window.
The old 60-observation rejection is finite-sample evidence, not a theorem about
all future quote states. This exception permits one disjoint diagnostic window,
not resampling until success, an old-result repair, account access, orders or
edge promotion. The contract's budget and terminal stop override this exception
after its one execution. All other literal retry/protected-capture rules remain.

## Hard Rules

- Apply the full [progress, blocking and resumption protocol](docs/AGENT_WORKFLOWS.md#progress-blocking-and-resumption)
  on every continuation. A blocked research branch is not a blocked repository.
  Exhaust safe, relevant alternatives before declaring a goal-wide impasse;
  do not manufacture progress with repeated scans, status commits or training
  on unchanged inadequate labels. Testnet success is not mainnet profitability.
  Use the three-consecutive-turn blocked audit only for a genuine impasse,
  reset that audit on user resumption, and never equate blocked with complete.

- For new make/take economic evaluations in this review session, use
  `make_take_forward_evaluation.evaluate_make_take_policy_forward` with the
  complete calibration and evaluation role inputs. The preserved Round 57
  evaluator alone does not reject overlapping role days or absent day-path
  evidence. Require calibration labels to end strictly before evaluation
  decisions; preserve the old runner, source bindings and historical results.

- Work in this session only; use no subagent.
- Apply the shared [codebase consistency standard](CONTRIBUTING.md#codebase-consistency-standard)
  through reasoned semantic and architectural review, not mechanical test or
  formatting compliance alone. Keep naming, comments, types, errors, logging
  and interfaces coherent; preserve frozen evidence and justified differences.
- Complete the user's [capital-protection and final review requirements](docs/CAPITAL_PROTECTION_ARCHITECTURE.md):
  independent process safeguards, durable recovery and model-health controls;
  after the main revamp, review every code line and make justified refactors and
  upgrades, then perform exhaustive final bug hunting. Track exact file/revision
  and line coverage. Documentation, syntax scans and passing tests alone do not
  prove this complete; preserve historical sources/results and reopen affected
  review coverage after subsequent changes.
- AI Git history must use `AI agent <>` for author and committer. Read
  `docs/AI_COMMIT_IDENTITY.md` before committing or auditing; never use a human,
  host, tool, CI, global-config, or noreply identity.
- Binance execution is BTC/ETH/SOL testnet, Demo, or paper only. Polymarket
  order-capable research remains BTC/ETH/SOL and its BTC-only live-capable
  boundary defaults off. Public unauthenticated read-only structural discovery
  may cover other Polymarket markets when it uses the same fail-closed payoff,
  source, and cost gates. No live-money authority exists.
- Conservative is default. Leverage is a ceiling, never edge. Profitability,
  ROI, readiness, and drawdown claims require source-bound after-cost evidence.
- Aggregate performance never establishes an all-regime edge. Apply
  `docs/model-research/cross-regime-edge-acceptance-contract-v1.json`; an
  unsupported bullish, bearish, sideways, choppy, volatility, liquidity, or
  latency slice must reject promotion or abstain from new exposure.
- Model promotion requires each objective's affirmative, noncontradictory
  selection-risk evidence. Missing status is not a pass; nonfinite performance
  is not an edge. The training suite's heuristic score haircut and two-panel
  overfitting proxy are not a statistical Deflated Sharpe Ratio or full
  combinatorial validation. Preserve old results and label these limits.
- Portfolio correlation and tail-risk evidence must align returns by both
  observation endpoints. Never substitute equally sized row tails for disjoint
  histories, match unequal return horizons only by their final timestamp, or
  replace nonfinite returns with zero. Duplicate endpoints are ambiguous and
  cannot establish diversification. Preserve historical reports; verify their
  alignment independently before treating them as current promotion evidence.
- Risk, ownership, reconciliation, Stop, and close controls are deterministic.
  AI may only veto or reduce risk after matched uplift evidence and may never
  override a safety gate or block a close.
- Polymarket terminal state requires authenticated exact-order or fill evidence.
  Stop may cancel and sell only bot-owned hashes and
  parent-bound lots; foreign state is never modified.
- Future books, labels, resolutions, fills, and PnL never enter inference.
  Full-fill support is not an inventory ledger: a censored incomplete label
  cannot establish zero partial fill or zero PnL, and equal quote notionals at
  different leg prices do not establish equal net base quantities. Before any
  forward maker economic replay, reconcile partial quantities, cash and residual
  exposure; preserve frozen historical support implementations and results.
  Unknown order or redemption state blocks new exposure. Polymarket settlement
  never auto-deploys wallets or creates token approvals.
- A public fill after an oracle deadline does not prove its maker order was
  created after finality. Any finalized-winner latency study must bind the exact
  undisputed non-ignore on-chain state before hypothetical or owned order entry;
  a bid resting before that state is directional exposure, not structural carry.
- Never print, prompt, log, serialize, test, document, or commit credentials,
  secrets, tokens, signed requests, or unredacted secret fields.
- Do not label the common `0.00010000` USD-M funding value as a funding-rate
  cap. It is normally the standard eight-hour interest-rate plateau produced by
  the premium-index clamp; bind any actual cap or floor from the applicable
  current `fundingInfo` terms before using cap semantics.
- Preserve testnet, dry-run, diversification, liquidity gating, and the app's
  `20x` leverage cap unless a stricter frozen contract applies.
- The installed CLI and native app both invoke `simple_ai_trading.entrypoint`.
  Register independent command extensions there and keep
  `command_contract.py` on the same parser; do not create frontend-only
  Polymarket controls.
- No network calls in tests unless explicitly stubbed. Do not hard-code host
  capabilities; detect and record effective backends and fallbacks.
- This workstation is shared with the user's other active tasks. Never stop,
  reprioritize, change affinity, or otherwise modify unrelated processes to
  improve a benchmark. Before performance-timing benchmarks, require
  a passive CPU/GPU/disk/memory headroom check that budgets our expected work;
  normal background activity and brief spikes are acceptable. Do not demand an
  almost-idle PC or treat total utilization below 100 percent as proof of no
  shared-core, cache, bandwidth, or storage contention. Record competing load
  during the run as well; a preflight alone does not validate later timings.
  Deferral is only for timing validity when sustained contention could materially
  distort a benchmark. Ordinary training and R&D may continue with bounded
  resource use alongside other tasks; do not impose an idle-window prerequisite
  on that work.
  Stop only verified task-owned
  benchmark processes if contention develops, preserve partial results, and
  label unmonitored prior timings provisional rather than clean speed evidence.
- For substantial new training, prefer a measured faster compatible accelerator,
  not a hardware-name assumption. Dependency upgrades can replace GPU-enabled
  libraries with CPU-only wheels: probe the actual installed backend and bind
  its library/runtime identity before large fits. Preserve each frozen model's
  backend and precision contract; synthetic speed is not financial uplift.
- Edge discovery is the current priority. Public, unauthenticated, read-only
  source, market-data, and blockchain research may proceed iteratively when
  each request tests a distinct question or materially refreshes stale evidence.
  Record request bounds and retain raw evidence, but do not require a commit,
  push, or hosted CI run before exploratory public requests. The stricter
  frozen one-use workflow remains mandatory for authenticated, account-specific,
  funded, order-capable, or state-changing operations.
- Before source selection, market-data access or any research capture, read
  [Research Capture Boundaries](docs/RESEARCH_CAPTURE_BOUNDARIES.md) completely.
  That mandatory companion preserves the full detailed capture/retry rules;
  do not treat its separation from this file as optional guidance or permission
  to reopen consumed studies. Relevant executable contracts remain controlling.

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
   For a documented cursor endpoint whose sampling cadence or total row count is
   not source-proved, freeze the conditional cursor traversal, maximum pages,
   total rows, total bytes, deduplication key, and fail-closed stop conditions
   before the first economic page. Size each page ceiling from the documented
   maximum row count times a conservative row-size bound rather than an
   arbitrary small guess. A consumed ceiling or pagination failure never
   authorizes a refetch or adaptive page plan; retain the exact bytes and perform
   at most the already-frozen zero-network adjudication.
5. Keep edits scoped, match existing patterns, remove only resulting orphans,
   and never revert unrelated work.
6. Keep numeric evidence in canonical JSON/CSV and regenerate charts from it.
   Generated charts and prose are not result authority.
7. Format code before generating implementation-hash-bound evidence; later
   source edits require regeneration.
8. Before freezing a one-use runner, execute its import-only or `--help`
   preflight through the same locked entry mode used for capture. In this
   repository, prefer `uv run python` so the `uv.lock` dependencies are present;
   never substitute the Windows Store `python` alias or an unverified global
   interpreter. Verify every required transport import before freezing the
   contract. For a retained-input runner, also parse every hash-bound raw input
   through the exact production loader and exercise the full pre-network
   contract validator before freezing; `--help` proves only imports and CLI
   wiring, not encoding or retained-schema compatibility. A runner that imports
   `tools.*` must be invoked as
   `uv run python -m tools.<module>` from the repository root; do not discover a
   runtime, dependency, or module-path difference after freezing evidence. If a
   pre-main import still fails, journal it as a local unconsumed preflight error
   before any corrected invocation.
9. Before adding a runner at a reusable-looking path, check path ownership with
   `git ls-files --error-unmatch <path>`. Never replace a tracked generic runner
   with a one-event implementation: pass it a new frozen contract or use a
   separately named wrapper. If replacement is discovered only after a request,
   do not rerun or rewrite the consumed contract; preserve the exact consumed
   implementation in an immutable hash-bound sidecar, restore the reusable path,
   and record both bindings in the terminal artifact.
10. Discovery and search surfaces may select an exact current primary source
    but never supply its economic values. If the one-use retained primary bytes
    disagree in currency, rate, scope, or formatting, preserve the failed
    contract and consumed response, exclude every discovery value, and perform
    at most one zero-network adjudication of the retained bytes. Never refetch,
    alias, or loosen the consumed contract to recover the expected answer.
    Freeze an exact required phrase only when that exact lexical form was
    already observed in the selected primary source. When discovery proves only
    a semantic proposition, use the smallest exact source-observed substrings or
    prospectively freeze every acceptable lexical alternative before access.
    Never guess a synonym, punctuation form, or locale variant as an exact gate.
    A missed phrase remains a failed one-use contract even when retained bytes
    later prove an equivalent proposition; do not rewrite, alias, or refetch it.
11. For recurring fixed-NegRisk weather or scalar partitions, freeze the exact
    deterministic series member, complete expected outcome count, and one-use
    source request before opening an event page or search result that can expose
    prices. If rendered prices leak first, mark that exact event permanently
    promotion-ineligible: it may be retained for rejection and lineage, but it
    cannot select books, authorize downstream requests, or support an edge
    claim. Route the next distinct series member prospectively instead of
    disguising or repairing the leak.

Do not broadly read the README, historical round designs, generated SVG, or
large CSV files. The detailed workflow and imported-tool provenance are in
`docs/AGENT_WORKFLOWS.md`; broad architecture starts with
`docs/SIMILAR_TRADING_REPOS_REVIEW.md`.

## Verification

- Invoke repository tests as `uv run python -m pytest`; the `uv run pytest`
  console entry may omit the repository root and cause false collection errors
  for tests that import `tools.*`.
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
