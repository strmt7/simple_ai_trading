# Continue Development

Read [AGENTS](../AGENTS.md), [Agent Start](AGENT_START.md) and the
[progress, blocking and resumption protocol](AGENT_WORKFLOWS.md#progress-blocking-and-resumption).
The full goal is active: profitable structural-edge R&D, model/data/execution and
capital-risk improvement, whole-app refactoring/visual quality, security and
packaging, followed by complete semantic review and exhaustive final bug hunting.
Neither partial verification nor an administrative checkpoint completes it.

## Current decision

Financial counts remain **203 observations / 65 hypotheses / 37 scoped mechanisms
/ 0 qualified stable edges**. Favor recurring net cash and sound payoff identities;
expiring promotions and larger datasets alone are not the objective. Use the full
[revamp work map](REVAMP_2026_11_09.md) and [work priorities](WORK_PRIORITIES.md).

Reverify HEAD/worktree and external state. The completed
[reserved-cash checkpoint](review/2026-11-09/spot-reserved-cash/review.md) has 41
focused checks and byte-identical reproduction of the old prefunding controls.
[Post-expiry settlement research](review/2026-11-09/settlement-equivalence/review.md)
remains source-incomplete. Each linked report retains its exact source phase.
There is no green whole-suite, qualified stable-edge or institutional-readiness
claim. The earlier dependency full-suite recheck retained 86 failures; one stale
readiness fixture was subsequently repaired in a focused lane. Do not infer a
current full-suite failure count without rerunning the appropriate scope.

## Financial and source frontier

- Ordered-fill spot cash must cover acquisition, quote fees and a declared buffer.
  Boundary net cash and terminal wealth can hide intermediate cash demand.
  `spot_prefunding.evaluate_hedged_spot_prefunding` keeps derivative income and
  opportunity charges separate from spot-wallet funding. Native cash provenance,
  locks, settlement delays, financing and continuous derivative margin remain open.
- [Reserved quote cash](review/2026-11-09/spot-reserved-cash/review.md) adds explicit
  maximum remaining locks at every fill prefix, including intervals with no fill.
  It preserves the old fill-only report and avoids double-charging partial fills.
  This is supplied-scenario accounting; native reserve coverage, settlement and
  continuous solvency remain unqualified. It neither releases live locks nor
  changes order authority. Polymarket already blocks unresolved-order reopening.
- Rank 13's new BTC September 25 settlement contract is consumed. The option API
  returned 100 rows, 44 same-strike call/put pairs and one exact settlement value.
  The 18-row futures response lacks that date; its latest marker is June 26.
  Zero numeric comparisons completed. No equality, mismatch, books or edge follows.
  Its updated missing-record/distinct-later-expiry gate controls; do not shift dates,
  paginate, alias, repeat the window or reopen existing box-spread rejections.
- Rank 59's [nine-week funding study](review/2026-11-09/weekly-funding/review.md)
  consumed its September 7-November 9 population when the first inventory GET
  exceeded its frozen 1 MiB body cap. HTTP 200 is not complete-source admission:
  1,048,577 partial bytes are retained; zero funding/index requests followed.
  No economic rejection follows. Do not enlarge the cap, alias, retry or request
  that population's funding/books. A later study needs its literal independent
  material-source or informative distinct-population trigger, not time alone.
- [Timed native replay](review/2026-11-09/timed-snapshot-normalization/review.md)
  fixes the actual HftBacktest time-zero snapshot leak. 64 capture/normalization
  and 20 readiness checks pass in overlapping lanes. EOF is not a valid clock step.
  The REST documentation crawl ended 302/202/empty; it is rejected and consumed.
  Declared clock units/offsets are not independently qualified Binance semantics.
- [Local snapshot timing](review/2026-11-09/snapshot-availability/review.md) gates
  inference/entries and preserves warmup. [Native depth](review/2026-11-09/depth-source/review.md)
  and [paired source](review/2026-11-09/paired-data-source/review.md) receipts retain
  one fixed minute and six late snapshots. Cross-collector clocks, continuous
  paired coverage, native costs and strategy replay remain unqualified.
- [Paired cash](review/2026-11-09/paired-inventory-cash/review.md) retains exact
  quantities, funding bounds, fees and non-atomic orphan ranges. The
  [term-premium memo](review/2026-11-09/funding-term-premium/review.md) is conditional
  algebra, not a margin saving. Its paper crawl failed 429; the second page stayed
  forbidden. Do not fit inadequate data, reprice consumed panels into independent
  validation, or bypass account gates with another carry variant.
- Polymarket remains open to eligible public/offline structural research, not
  unrestricted account/order access. Match exact settlement/payoff identity,
  token/lot ownership and costs. Protected partial captures, all consumed windows
  and their literal retry triggers remain unchanged. No blanket venue-wide block
  and no repeated polling of unchanged promotional programs is justified.

Next choose a distinct informative source/economic question whose complete retry
trigger is independently satisfied, or close a demonstrated capital/recovery gap.
Qualify causal native units/clocks, paired quantities, every cost and feasible
capital before expanding fits. Actual backend/precision, useful dataset coverage
and measured bottlenecks determine GPU work; do not disturb other host tasks.
LLMs may veto/reduce risk, never bypass deterministic Stop/ownership/close controls.

## Platform work and publication

- [Atomic order admission](review/2026-11-09/order-admission/review.md) fixes a
  reproduced Polymarket stale-preflight race: a pending order created during
  another order's signing can no longer admit a second opening. The same SQLite
  writer transaction checks unresolved orders and inserts the reservation;
  separate connections admit one opening. 240 focused lifecycle/runtime checks
  pass. Closing gates remain separate; complete capital/supervisor qualification
  and races involving completed fills or changing risk limits remain open.
- [Source-bound navigation](review/2026-11-09/semantic-routing/review.md) now fuses
  native CocoIndex semantics with lexical ranks from the same approved index.
  The frozen paired check improved expected-file hits from 0/10 to 7/10 on
  observed cases and 3/5 to 4/5 on fixed follow-up questions; four misses remain.
  167 focused checks and real MCP probes pass. This is developer evidence, not
  independent validation, financial evidence or a timing claim. Search never
  builds a cache or falls back after native failure. Preserve native-only mode,
  linked receipts and historical source phases; return to financial/capital work.
- [Dependency/publication review](review/2026-11-09/dependencies/review.md): latest
  compatible locks and owned automatic installation are verified in their stated
  scopes. Readback at 16:57 UTC November 9 confirmed all 17 audited Dependabot
  alerts fixed, no open alerts/PRs and no public releases/tags. Other hosted
  quality findings, comprehensive security acceptance and final review remain open.
- Normal commits use only the official GitHub connector's assigned attribution
  and expected-head leases; see [identity rules](AI_COMMIT_IDENTITY.md). The
  authorized historical cleanup preserves every tree and original result identity
  via its [commit map](review/2026-11-09/history-attribution/commit-map.json).
  No further rewrite, release, tag or extra PR follows from a continuation.
- [Storage relocation](review/2026-11-09/research-storage/review.md) verified 6,783
  files on SSD and retired the authorized old HDD directory. Do not reread the
  large dataset or restart migration. Private paths/mappings stay ignored.
  Automatic review blocked unused temporary-directory removal; files remain.
  Do not bypass that rejection or install a cleanup worker.
- [Operator view](review/2026-11-09/operator-overview/review.md) improves typed
  presentation and bounded hidden rendering. Other pages, real backend integration,
  diagnostics, persistent capital limits, supervisor/recovery, model admission,
  distribution protection and whole-tree review remain in the revamp scope.
  Hidden simulated renders are not interaction or live-trading evidence.
- No current capture, training, benchmark or development worker is authorized to
  persist. Unknown MCP-client ownership is not permission to kill other tasks.
  Keep host/install paths configurable and dependencies private. Batch meaningful
  checkpoints, verify changed boundaries and avoid repetitive full-suite reruns.

## Preserved history

The [archive index](archive/agent-history/README.md) binds all 17,028 prior handoff
lines to their original published blobs, hashes and revision. Original relative
references resolve from `docs/`; use immutable permalinks for rendered historical
links. Non-superseded study constraints remain in force. Current user instructions,
AGENTS, exact canonical contracts and the current registry control conflicts;
absence from this compact summary never creates access or promotion authority.
Do not rewrite old evidence to match current code or cleaned-up history.

Maintain one current summary per workstream here. Put detailed new results in
source-bound review artifacts and link them instead of appending dated transcripts.
