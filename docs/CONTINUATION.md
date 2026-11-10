# Continue Development

Read [AGENTS](../AGENTS.md), [Agent Start](AGENT_START.md) and the
[progress, blocking and resumption protocol](AGENT_WORKFLOWS.md#progress-blocking-and-resumption).
The full goal is active: profitable structural-edge R&D, model/data/execution and
capital-risk improvement, whole-app refactoring/visual quality, security and
packaging, followed by complete semantic review and exhaustive final bug hunting.
Neither partial verification nor an administrative checkpoint completes it.

## Current decision

Financial counts are **204 observations / 65 hypotheses / 37 scoped mechanisms
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

- [Spot protocol sources](review/2026-11-09/native-clock-source/review.md): two
  immutable official revisions passed native source admission. Both document
  default milliseconds and optional microseconds; Spot bookTicker has no native
  event-time field. [Vendor collection evidence](review/2026-11-09/collector-clock/review.md)
  documents microsecond Spot capture since February 28, 2025; use that declared
  vendor mode for the October sample, conditional on its retrospective claim.
  Arrival precision is not accuracy; actual historical deployment and clock
  offsets remain independently unverified. The vendor's historical mapper may
  suppress snapshot-overlap errors. Preserve the initial/subsequent bridge
  boundary distinction; the futures `T`/`pu` normalizer is not a Spot adapter.
  Its clock probe now rejects malformed clocks and discontinuities and preserves
  small integer differences; 86 scoped checks pass, without clock qualification.
  The [retained clock audit](review/2026-11-09/spot-clock-audit/review.md) completed
  once: 1,755 depth events, no E regression and observed receipt-minus-E ranges
  of 1.6642-17.5574 ms across symbols. These are cross-clock differences, not
  measured latency. All three snapshots lack E/T. Preserve nullable snapshot
  exchange time and actual collector availability in a future causal Spot path;
  native-clock replay, full coverage and paired costs remain open. No new market
  download or economic observation followed, and old window restrictions stand.
  Routine public technical-document decisions now use the user's explicit
  session delegation in AGENTS, without repeated approval requests. Earlier
  source failures, consumed financial tests and protected captures stay intact.
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
  The separate [2025 annual diagnostic](review/2026-11-09/year-funding/review.md)
  completed 46 native responses, 36 monthly cash intervals and 6/3/3-month roles.
  BTC/ETH have positive conditional at-par funding cash in all twelve months,
  but all nine role nets fail the fixed reserves and explicit two-leg capital
  hurdle; SOL also fails late persistence. Zero survivors; this population is
  consumed. All 36 cash rows and nine capital/net identities were reconciled.
  Historical terms/index continuity and actual total hedged profit remain
  unqualified. No reprice, subset, reorientation or book/account follow-up;
  use the unchanged literal trigger for an informative distinct future study.
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
- [Finality and cash](review/2026-11-10/finality-cash/review.md): two native primary
  paper captures passed. First-redemption-by-any-holder timing cannot qualify this
  bot's usable collateral clock or turnover. The source cohort predates the August
  rank-6 screen; its intended post-oracle/pre-adapter probe window and exact retry
  trigger remain unchanged. Holder/payout lineage and effective/available clocks
  now have explicit evidence requirements. No market request, fit or edge followed.

Next choose a distinct informative source/economic question whose complete retry
trigger is independently satisfied, or close a demonstrated capital/recovery gap.
Qualify causal native units/clocks, paired quantities, every cost and feasible
capital before expanding fits. Actual backend/precision, useful dataset coverage
and measured bottlenecks determine GPU work; do not disturb other host tasks.
LLMs may veto/reduce risk, never bypass deterministic Stop/ownership/close controls.

## Platform work and publication

- [Model selection integrity](review/2026-11-10/selection-risk-integrity/review.md)
  repairs two reproduced fail-to-pass paths caused by copying losing candidates.
  Search penalties no longer treat candidate count as time-series sample size;
  identical panel profiles cannot reweight the rank proxy, and ties are handled
  conservatively. Malformed explicit counts reject; valid large integer counts
  remain exact. All 124 final affected checks pass; six original-source
  regressions fail. The guard explicitly remains heuristic, with no significance
  or profitability qualification. Current author metadata was admitted, while
  three article captures failed their frozen gates. Full methodological text,
  return-path/search provenance and calibrated inference remain open; no expanded
  production fit or market resampling followed.
- [Binance scope enrollment](review/2026-11-10/binance-scope-enrollment/review.md)
  closes a reproduced first-binding gap that could assign unscoped legacy live
  inventory to new credentials. Version 3 binds paired bytes, scope and UNKNOWN
  atomically; new openings reject changed/lost enrollment. Bound version 2 retains
  recovery/owned closing, while new exposure requires verified migration. The
  passing 154/470/27-case lanes overlap; three original-source regressions fail.
  Complete migration, database-loss protection and universal fencing remain open.
  [Spot quote-budget documentation](review/2026-11-10/spot-quote-budget-source/review.md)
  now has an admitted immutable publisher copy; the first HTML shell remains
  rejected. It informs the roundtrip's outstanding operation-cash/two-leg design,
  without changing transport or qualifying any edge.
- [Owned Binance CLI execution](review/2026-11-10/binance-owned-cli/review.md) now
  shares the autonomous write-ahead opening/closing flow and returns typed raw-ack
  outcomes. Entry, normal and emergency close bind scope before dispatch; real
  partial residuals survive. 217 distinct focused cases are supported across
  documented lanes, with three old-CLI regressions. The Spot roundtrip's two
  direct writes, other native transports, legacy migration/rearm and independent
  process fencing remain open; there is no universal gateway or whole-suite claim.
- [Owned SDK transport](review/2026-11-10/sdk-transport/review.md) binds native
  Polymarket GET/POST/DELETE hooks to each venue's configured timeout and body cap,
  preserving signing bytes, no write retries and no redirects. Incomplete/late
  error bodies remain unresolved; cleanup releases owned clients. All 187 focused
  checks pass, including actual SDK serialization and an owned loopback response.
  Supplied external clients and hard process deadlines remain unqualified.
  Independent supervision, gateway fencing and the full caller inventory stay open.
- [Control storage](review/2026-11-10/runtime-control-storage/review.md) no longer
  recreates a missing current/legacy record as fresh stopped state. Bootstrap is
  atomic; cancellation rolls back and closes connections; a completed concurrent
  migration is verified. Seven old-source regressions fail; all 61 combined
  runtime checks pass. Complete namespace/database loss still needs an independent
  storage-identity and recovery contract. Order-caller inventory and independent
  trading supervision remain incomplete; no native account/order was used.
- [Runtime interlock cleanup](review/2026-11-09/interlock-cleanup/review.md) repairs
  leaked file handles after setup/unlock failure and a held mutex after tested
  acquisition cancellation. Three old-code regressions fail; 29 scoped checks
  pass, including real-file peer reacquisition and existing Stop/close ordering.
  The capital blueprint now explicitly scopes the supervisor to app sessions,
  without persistent service/autostart installation. Independent trading-process
  supervision, bounded native calls, durable policy and reboot proof remain open.
- [Capital admission](review/2026-11-09/capital-admission/review.md) extends the
  [pending-order fix](review/2026-11-09/order-admission/review.md): openings bind
  the preflight-start ledger revision and captured risk limits through admission;
  completed fills or policy changes during signing reject the stale decision.
  Close capacity is rechecked against its exact parent in the reservation's writer
  transaction; unrelated activity does not veto a valid close. Lot reads use one
  SQLite snapshot. 248 focused lifecycle/runtime checks pass. Native balances,
  cross-process policy governance, supervisor/recovery and full capital acceptance
  remain open; no market-model or profitability qualification follows.
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
