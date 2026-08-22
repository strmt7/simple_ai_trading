# Agent Start

This is the smallest safe entry point for work in this repository. It routes
agents to canonical evidence without replacing that evidence.

## Non-negotiable truth

- Binance scope is BTC, ETH, and SOL on testnet/Demo or paper only. Polymarket
  research covers BTC/ETH/SOL; its independent live-capable boundary is
  BTC-only and disabled by default. No model or release has live-money
  authority.
- Conservative is the default profile. Leverage is a risk ceiling, never a
  source of edge. Profitability, ROI, and drawdown claims require reproducible
  source-bound after-cost evidence.
- Risk, reconciliation, Stop, and ownership checks are deterministic. AI may
  veto or downsize only after matched uplift evidence and may never block a
  close or override a safety gate.
- Polymarket terminal state requires authenticated exact-ID order evidence or
  exact fill evidence. Stop cancels only owned hashes and closes only
  parent-bound, confirmed, unreserved lots from fresh Polymarket books.
- AI uplift v3 requires at least 30 non-tied matched outcomes. Exact ties are
  excluded from the paired sign test but retained in the moving-block bootstrap
  and the contiguous 90-day coverage evidence.
- Profit factor is gross profit divided by gross loss and is capped at `999.0`
  in finite JSON evidence; positive P&L with zero gross loss uses that cap.
- Return-to-drawdown evidence is bounded to `[-999.0, 999.0]`; positive return
  with zero drawdown uses `999.0` instead of being misreported as zero quality.
- Publication independently reconciles configured and reported capital, filled
  trade P&L, equity, drawdown, and both initial- and deployed-capital returns.
- Round 9 action labels independently reconcile fill quantities, weighted fill
  prices, per-fill fees, entry cost, exit proceeds, and net value before storage.
- Historical labels, future books, resolutions, fills, and PnL must never enter
  a live inference payload. Unknown order or redemption state blocks new
  exposure. Polymarket settlement never auto-deploys wallets or creates token
  approvals. A successful redemption transaction is not realized payout
  evidence: the canonical finalized Polygon receipt must bind the condition,
  adapter, dedicated wallet, pUSD wrap, mint, and one exact payout amount.
- Secrets must never enter prompts, logs, artifacts, tests, commits, or docs.

## Task routing

| Task | Read first | Canonical evidence |
|---|---|---|
| Binance model or backtest | nearest model module and test | `docs/model-research/action-value/latest/README.md`, then selected rows from `progress.csv` |
| Prior model failure | last row plus the relevant mechanism row in `docs/model-research/action-value/latest/progress.csv` | that row's named design/report only |
| Polymarket model | Latest canonical status plus the target-blind Round 21 finite model, economic replay, and AI candidate-selection contracts | `docs/model-research/polymarket/latest/README.md`, then the Round 21 contracts |
| Polymarket recorder/replay | matching recorder or replay module and test | `docs/model-research/polymarket/prospective-continuity-contract-v2.json` |
| Risk or execution | nearest risk/execution module and test | `docs/LIVE_MARKET_SIMULATION.md`, `docs/POLYMARKET_PAPER_TRADING.md`, or `docs/POLYMARKET_LIVE_EXECUTION.md` only at the relevant heading |
| AI provider/model | nearest AI module and test | `docs/ai/risk-review/latest/comparison.json` and its sibling provenance |
| CLI | command handler, parser definition, and CLI tests | parser-generated help; do not infer parity from docs |
| Windows app | `src/simple_ai_trading/command_contract.py`, `src/simple_ai_trading/windows_app.py`, and the UI/parity tests | `native/windows/generated/command_contract.hpp` and `tests/test_ai_runtime_and_parity.py`; edit the shared taxonomy, never the generated header |
| CI/release | one workflow and its test/lint config | `docs/AGENT_WORKFLOWS.md` |
| Broad architecture | `docs/SIMILAR_TRADING_REPOS_REVIEW.md` | source and tests for each affected boundary |

The native app verifies the backend's `ui_contract` SHA-256 before any ordinary
workflow. A mismatch blocks Start and expert commands but never Pause or Stop.
After parser or taxonomy changes, regenerate the header through the native build.
The installed CLI and native app invoke `simple_ai_trading.entrypoint`; it
extends the established parser with independently registered commands.
`command_contract.py` must inspect that same entry point.

## Model research state

- The compact cross-round ledger is
  `docs/model-research/action-value/latest/progress.csv`. Read its header and
  only the last few rows unless a task names an older mechanism.
- **Local 2026-08-22 integration boundary:** pushed `main` does not contain the
  active Round 75 implementation. That capture is running from a detached,
  materially dirty local worktree whose HEAD must not change before frozen
  terminalization. The independent Round 21 Binance sidecar is also active.
  Read `docs/CONTINUATION.md` for exact processes, state hashes, scheduled end
  times, security coverage, and the post-capture integration sequence. Never
  clean, reset, stage, commit, switch, or merge either live capture worktree.
- **Current capture boundary:** Round 73 v9 is invalidated before target or
  model access. Eighteen indexed hours passed the old gate, but two later
  reconnect-free, exact-audit-passing hours were rejected solely because the
  process-I/O counter was divided by lower market-message counts. Continuing
  would select higher-activity hours. Never resume v9, pool its 18 hours, or
  use them for a model or profitability claim. Read
  `round-073-v9-corpus-invalidation-2026-07-25.json`. Round 74/v10 is the only
  admissible capture-schema lineage, but no Round 74 campaign is currently
  open. It uses separate tables and hashes, provider-driven heartbeat handling,
  independent data/resource verdicts, and absolute activity-independent host
  ceilings. Read
  `round-074-capture-recovery-design-v1.json` and
  `round-074-capture-contract-v10.json` before touching capture code.
- Round 72 rejected aggregate spot/perpetual price discovery: all 9 components
  and all 36 incremental comparisons failed their frozen gates. Round 73 is a
  prospective multi-level impact-absorption hypothesis, not a model result.
  Contracts v1-v7 preserve the feed corrections and failed storage experiments;
  never pool disconnected attempts or reinterpret them as model evidence.
  Contract v8 routes new exact frames and typed streams to isolated versioned
  tables while keeping v1-v7 audits reproducible. One-hour run
  `f3e92ba29e1e4d3188c3f309f5c160a2` passed its capture gate with 1,294,128
  messages, 847 frames, zero reconnects, zero physical database growth, 21.68%
  peak queue use, and 3,514.6 process-I/O bytes per message against the frozen
  4,096 limit. A fresh process audited every frame. Independent exact-wire
  replay reconciled all 104,305 depth-band rows and reconstructed 4,459,493
  level changes without future data. Read `round-073-capture-contract-v8.json`
  and `round-073-v8-capture-qualification-2026-07-22.json`. V9 retains exact
  frames and low-rate REST context, removes duplicate live typed projections,
  and passed its 180-second public-feed gate in run
  `c096c88375e24bfdba560c7f32f8a121`. Its first one-hour attempt
  `bc032079846b40b58fbcfe8786afab64` is permanently failed and excluded: the
  writer connection inherited the 16 MiB default instead of the persisted
  512 MiB run policy. All three segments are invalid, the fresh exact audit
  passes, no corpus manifest exists, and the WAL recovered cleanly. The writer
  now binds and verifies the persisted policy before readiness and before every
  frame append; the affected 138-test Round 73 checkpoint passes. Read
  `round-073-v9-one-hour-qualification-failure-2026-07-23.json`. Immediate retry
  `676d219ba329445f85645b2fae50a60f` is also permanently failed and excluded:
  an ad hoc PowerShell wrapper buffered progress and monitored the wrong WAL
  filename, so the operator stopped an otherwise unqualified run. Its 138,216
  messages pass exact audit, but all segments are invalid and no corpus manifest
  exists. The CLI now selects v9 directly, reports the correct `.duckdb.wal`
  bytes, and remains synchronized with the generated Windows contract. Read
  `round-073-v9-qualification-operator-abort-2026-07-23.json`. Replacement run
  `0aabddb515794668a8a54129aa6e1d47` then passed the one-hour v9 qualification:
  2,277,593 real public Binance messages, 856 frames, zero reconnects, zero
  negative corrected latency, 5.16% peak queue use, and 79.07 process-I/O bytes
  per message. A fresh process audited the entire chain. Its first downstream
  replay failed closed because buffered public depth receipts legitimately
  preceded REST snapshot records; this did not revoke capture qualification.
  V3 corpus/grid contracts now preload the immutable snapshots only for state,
  apply pre-ready depth only for sequence continuity, and exclude every
  pre-ready receipt from feature aggregates. The repaired v4 replay passed all
  2,277,593 messages: 123 of 104,385 depth updates were pre-ready, including 37
  valid sequence advances, and 104,262 were feature-eligible. The affected
  125-test checkpoint and Ruff pass. Read the v9 qualification, preflight
  failure, and feature-source-success artifacts dated 2026-07-23. At that
  success artifact, no v3 corpus or grid row existed. Multi-segment capture,
  model evaluation, P&L, profitability, AI uplift, leverage, and all trading
  authority remain closed.
  The subsequent v3 corpus manifest passed its independent audit, but the first
  v3 grid is permanently rejected. Its post-write audit found impossible
  rolling values; a full read-only scan found at least one financial invariant
  failure in 10,538 of 10,619 vectors. Repeated binary64 addition/subtraction
  had left cancellation residuals in nonnegative quote, depth-flow, and
  liquidation totals, producing buyer-share values from -8192 to 64.34. Keep
  all v3 grid rows as failure evidence only. V4 uses compensated nonnegative
  totals, exact zero reset only when no nonzero term remains, and shared
  pre-write/post-write vector invariants; it never clips residuals. The one
  authorized replacement v4 grid then passed its independent persisted audit
  and a separate full financial scan: 10,619 valid vectors, zero financial or
  anchor-primitive violations, bounded shares in `[0, 1]`, and no negative
  nonnegative-class values. A rejected diagnostic had incorrectly treated
  normalized order-flow turnover as a share; all 222,999 values instead
  reconciled exactly to signed cumulative displayed-depth flow divided by
  contemporaneous displayed depth. Read
  `round-073-v3-grid-numerical-failure-2026-07-23.json`,
  `round-073-causal-grid-contract-v4.json`, and
  `round-073-v4-grid-qualification-2026-07-23.json`. V3 remains preserved and
  excluded. The executable-target v1 contract was frozen before target replay.
  It materializes every valid anchor before training-fold shock selection,
  quantizes symmetric long/short quantities from causal decision prices, walks
  only observed top-20 depth after 500 ms and 1,000 ms delays, applies a 12 bps
  minimum round-trip fee/adverse reserve, and fails closed on late states,
  insufficient capacity, invalid exchange filters, and funding boundaries.
  Read `round-073-executable-target-contract-v1.json`. One target-mechanics
  replay is open; model evaluation and every profitability or trading claim
  remain closed. That replay is now consumed and independently audited: 380,483
  of 382,284 options were mechanically eligible, but none cleared the frozen
  12 bps round-trip reserve. Best gross paths were 8.08 bps BTC, 11.52 bps ETH,
  and 10.29 bps SOL. This validates target mechanics but leaves a single-class
  binary target, so no model was trained. The observed hour is development-only
  and excluded from future selection or scoring. Do not weaken the cost, delay,
  horizon, or population after this result. Read
  `round-073-v1-target-mechanics-diagnostic-2026-07-23.json`. A compact v2
  contract is now frozen in `round-073-compact-shock-target-contract-v2.json`.
  It admits untouched v9 anchors from 2026-07-24 UTC, freezes a 4/1/2-day
  train/tune/test split, derives shock thresholds from training features only,
  and permits target replay only after the selected cohort is hash-bound. The
  deterministic cohort builder and deep auditor are implemented, including
  source-overlap, pre-labeled-source, threshold, embargo, refractory, and hash
  checks. The selected-anchor v2 replay, per-run atomic manifests, row/source
  audits, zero-anchor handling, and all-source study seal were implemented as
  mechanics, but their all-role sequence is superseded for eligible data.
  Each audit independently replays its source's exact-wire frames and requires
  byte-equivalent target rows. They inherit the v1 book walk, quantity, funding,
  path-risk, and 12 bps charge logic while changing only the preregistered
  anchor population and 15/60/300 second horizons. No eligible seven-day cohort
  or v2 target result exists. The v2 builder now rejects every eligible anchor.
  `round-073-staged-holdout-contract-v4.json` requires role-scoped development
  replay, an immutable pretest model/policy manifest, a one-time test unlock,
  and test-only replay. The v3 target store, byte-persisted pretest artifacts,
  clean-Git identity check, fitting-row reconciliation, one-time unlock, test
  seal, and role-scoped operational dataset loader are implemented. Focused
  tests prove that no test target exists before unlock and that target,
  artifact, repository, or role drift fails closed. The bounded symbol loader,
  shallow model family, actual OpenCL device pinning, single-audit training
  publication, append-only access/prediction/result tables, all-scenario
  evaluator, and terminal interruption path are implemented and focused tests
  pass. No eligible target or model result exists. Held-out outcomes must not be
  queried before the immutable pretest publication and one-time unlock. The
  evaluation is separately frozen in
  `round-073-selected-anchor-evaluation-contract-v1.json`. Only deterministic
  source-boundary censoring may be removed. Pre-entry safety aborts remain
  attempted zero-return actions, while any selected unresolved post-entry exit
  fails that symbol's economic gate. The primary simulation is fixed at 500 ms,
  60 seconds, `$1,000`, `1x`, no reinvestment, and one open position per symbol.
  Models, transforms, tuning thresholds, and the action rule must be hash-bound
  before test rows are read. The seven-day result cannot authorize annualized
  claims, AI, leverage, or trading. The first transform is now frozen and
  implemented in `round-073-action-aligned-feature-contract-v1.json` and
  `impact_absorption_model_features.py`: it creates symmetric long/short views
  and strictly nested 90/107/261-feature layers without targets, fitted
  statistics, clipping, or invented ratios. This is feature plumbing, not a
  trained model or edge result. `impact_absorption_model_dataset.py` now adapts
  both pre-eligibility fixtures and explicitly staged v3 development/test rows
  into immutable, hash-bound action datasets while enforcing the frozen
  censor/abort/unresolved-exit semantics. `impact_absorption_model_slice.py`,
  `impact_absorption_training.py`, and
  `impact_absorption_one_use_evaluation.py` now complete the bounded fit and
  one-use evaluation path. Their tests use contract fixtures only; no
  prospective model dataset or result has been produced.
  The generated CLI/native workflow now orders this path as role-target stage,
  model fit, acknowledged test unlock, test-only stage, test seal, and
  acknowledged one-use evaluation. Both irreversible commands require explicit
  flags, every long phase emits progress, and the command-contract parity test
  fails if either frontend drifts.
  Evaluation now replays every claimed test source from exact wire before
  scoring, applies cumulative staged model gates, orders chronological folds by
  first market-time occurrence, requires paired block-bootstrap lower bounds,
  bounds scenario allocations, and distinguishes real per-position excursions
  from realized-exit portfolio drawdown. Viability requires at least two
  symbols to pass predictive, operational, and economic gates independently as
  well as a passing combined portfolio. Tuning now enables an action threshold
  only when its bootstrap lower expectancy is strictly positive, reserves both
  execution-lateness budgets in its overlap guard, and leaves an unresolved
  symbol occupied for the rest of tuning.
  This historical implementation remains auditable, but prospective v9
  collection is closed by the hash-bound Round 73 invalidation. Model
  evaluation and all trading authority remain closed.
  One hour is not a model-evaluation corpus.
  The segmented-corpus and rotation-runner contracts are now frozen. Historical
  runner v1 rows remain independently auditable, while runner v2 admitted v9
  capture, reports, and recovery only. It uses one lease owner, terminal
  batch journals, zero reconnects, one-hour segments, recovery-before-capture,
  and serial exact replay after capture.
  Qualification batch `ca83202743254d7ebc0c2d42d27d9b12` and run
  `cdac34967698498fbe27cfb299230fa8` completed one accepted hour with 857
  frames and 1,596,509 real public messages. Independent deep batch and run
  audits passed with zero errors; all three symbols had zero invalid events,
  sequence gaps, and crossed books. Physical database growth was 155,451,392
  bytes, proving the old 8 GiB cap has only about 16.9 measured-rate hours left.
  The planned seven-day collection cap was 48 GiB, with per-segment cap
  enforcement. That authorization is revoked because the later v9 campaign
  exposed activity-conditioned admission. Read
  `round-073-v9-qualification-capture-2026-07-23.json`; the run started before
  the July 24 prospective boundary and is qualification-only, not model data.
  Recovery-only batch `6d8c31559bb044b3a83fdf9e771dda4a` passed its real
  lease, discovery, terminal-journal, release, and independent audit paths with
  zero database growth. The required live runner segment and its deep batch
  audit passed as documented above; this does not authorize model evaluation or
  any further v9 collection.
- Round 61 rejected elevated-funding spot/perpetual carry on capacity, median
  after-cost return, and lower-confidence-bound gates. Do not tune or retrain
  that family.
- Polymarket Round 13 failed before outcome access. Its one-use prospective
  capture stopped after `1921.322` of the required `86400` seconds with
  `1281245` persisted source messages and `4` stream gaps. The terminal
  integrity audit was incomplete, normalized event materialization never
  started, and no labels, model scores, or outcome endpoints were opened. It is
  neither model nor performance evidence and can never acquire paper or live
  authority. Any successor must use a new preregistered contract and untouched
  capture; never resume, pool, or reinterpret the failed Round 13 prefix.
  Round 12 is invalidated before
  outcome access and is neither model nor performance evidence. Round 11 is an
  older rejected scored result. It reused the real
  47-group Round 9 corpus for development only, modeled one FOK entry held to
  resolution, and evaluated 42 chronological validation markets. The selected
  simulated point estimate was `+22.44105` quote across 42 displayed-book fills,
  but maximum drawdown was
  `12.36399` and the bootstrap lower mean-group utility was `-1.38152`. The
  learned external-feature residual norm was only `0.00117`; most apparent
  probability uplift came from recalibrating the market prior. No profitability,
  ROI, acceptable-drawdown, AI, paper, or trading authority exists. Read
  `docs/model-research/polymarket/latest/README.md`, then the Round 13 contract,
  Round 12 invalidation, and Round 11 report. Round 10 rejected the one-second scalp on negative action scores;
  Round 9 remains the immutable unknown-state admission failure.
- Polymarket Round 14's one-hour shadow rejected its first-candidate policy: 3
  wins, 9 losses, `-9.87720` counterfactual after-cost quote PnL, `0.455725`
  profit factor, and `11.69847` quote maximum drawdown. It is historical
  diagnostic evidence only. Round 20 is the active successor corpus: a
  target-blind BTC five-minute campaign scheduled from 2026-07-30 23:40 UTC
  through 2026-08-29 23:40 UTC under plan `2c1d8757...`. Its reboot recovery
  preserved the interrupted segment and started a new attested segment. One
  single-lane reconnect gap is recorded; the frozen contract rejects affected
  conditions after redundant-union reconstruction and does not silently treat
  the interval as continuous. No outcomes or model scores may be consulted
  during capture. Read the Round 20 contract and campaign design before touching
  this lane.
- Polymarket Round 15 is the preregistered expanded BTC five-minute historical
  screen. It binds 154 complete UTC days of official one-second BTCUSDT spot
  and USD-M perpetual archives, excludes all four previously consulted
  Round 14 target days, and reserves 15 untouched July days for one-use test
  access. Its fixed eight decision offsets must be selected on tune data only;
  never use the failed one-hour shadow to choose an offset. Round 15 is
  predictive research only and cannot establish execution, fill, latency, PnL,
  or live authority.
  Keep the actual fixed-resolution binary settlement label; do not substitute
  a continuous-market triple-barrier target. A 2026 nine-fold walk-forward
  BTC five-minute study found that relabeling systematically degraded the
  settlement target rather than improving it.
  `tools/ingest_polymarket_btc_history.py run` atomically republishes the
  self-hashed, non-authoritative ingestion status after closing the database;
  `publish-status` reconstructs it read-only after an interrupted publication.
- Polymarket Round 16 is a separately preregistered BTC fifteen-minute horizon
  comparison motivated by `arXiv:2606.31675`, which reports settlement-time
  manipulation in five-minute BTC contracts and substantially less in
  fifteen-minute contracts. It reuses the certified Round 15 Binance feature
  facts without duplicating them, but has separate Polymarket identities,
  targets, pretest, and holdout. No eligible Round 16 identity or terminal
  target was accessed before contract
  `6037c9ef473bcc736dbc7c3e98db76b75170e69e23de9574373bad7ae3fcdb67`
  was frozen. This v2 contract preserves v1 as an immutable predecessor and
  adds a moneyness-only financial control before any target access. It binds
  label-blind tune-only settlement-anomaly thresholds
  and train-only feature-support abstention before held-out access. Neither
  horizon may be pooled, selected, or promoted without its own prediction and
  prospective after-cost gates.
- Every Polymarket lane is execution-independent from Binance. Binance may
  supply optional public BTCUSDT predictor observations, but no Binance
  credential, account, balance, order, fill, position, capital, PnL, risk, or
  execution state may enter the Polymarket wallet, ownership ledger,
  reconciliation, stop, or settlement boundary.
- Round 21 adds only an optional, credential-free Binance spot/USD-M predictor
  sidecar. Its campaign, database, terminal identity, failure handling, and
  availability are separate from the Round 20 Polymarket corpus. Terminal
  development and one-use evaluation are frozen by
  `round-021-terminal-sealed-evaluation-design-v1.json`. Before test access, the
  manifest binds the clean repository, terminal captures, selected model layer,
  81-ledger development matrix, test population, and any development-nominated
  AI identity. The durable claim is consumed before test feature, target, or
  execution access and cannot reopen after completion, failure, or interruption.
  Terminal opening first uses
  `round-021-terminal-transport-manifest-design-v1.json`: it never opens the
  active database, preserves every failed/interrupted segment and scheduled
  coverage hole, and grants only permission for later exact receipt and
  redundant-union reconstruction. It is not condition or model eligibility.
  `round-021-core-corpus-materialization-design-v1.json` then freezes the
  single-pass condition boundary: per-segment union reset, mid-epoch Chainlink
  sequence binding without renumbering, exact open/close transport checks,
  whole-condition roles, joint-gap rejection, and all 1,200 causal decision
  times. Its implementation has no terminal corpus or result while capture is
  active.
  `polymarket_round21_corpus_store.py` stages separate development and sealed
  test DuckDB artifacts under one atomic directory publication. It stores no
  raw payload copies, uses one Zstandard level-3 exact-binary64 chunk per
  condition, and publishes nothing after a failed terminal audit. The pretest
  v2 seal derives its corpus and sealed-population hashes from that validated
  publication; sealed feature audits require the exact consumed claim and
  access hash in the durable one-use ledger.
  The target-blind model program is now frozen at six candidates per matched
  ledger: three logistic residuals, two shallow LightGBM residuals, and one
  compact causal TCN residual. The TCN uses 16 causal 250 ms rows, resets on a
  condition or cadence gap, samples eight endpoints per training condition,
  stores exact checksummed float32 weights, and reports every epoch. Do not add
  another candidate or interpret the successful AMD DirectML runtime probe as
  predictive or economic evidence. Candidate uncertainty is dependence-aware:
  ordered condition losses use Bartlett Newey-West standard errors, paired
  intervals use a circular block bootstrap, and the one-standard-error rule is
  computed from each candidate-minus-best paired loss series. The frozen block
  length is `min(n, max(2, ceil(sqrt(n))))`; do not replace it after targets are
  visible.
  No Round 21 claim, model score, economic verdict, profitability claim, or
  trading authority exists while capture remains active.
  The target-free prospective scorer and in-memory live-feature coordinator are
  implemented as non-authoritative prerequisites. They require an accepted
  sealed result, preserve the 16-row 250 ms sequence contract, interlock the
  current market after any Polymarket CLOB/Chainlink gap, and reset optional
  Binance state independently. The credential-free public session opens two
  redundant CLOB market channels and one Polymarket RTDS Chainlink channel in
  memory. Its strict RTDS classifier sequence-accounts the current empty
  opening frame and subscription-history snapshot but excludes both from
  causal features. A 2026-08-02 live-host transport probe processed 19,147
  frames in 15 seconds (19,131 CLOB, 14 live Chainlink updates, two RTDS
  controls), with zero gaps and a queue high-watermark of two. A separate
  three-second real-coordinator probe processed 2,106 frames into 243
  redundant-union events and three live Chainlink prices, also with zero gaps
  or core interlocks. Both probes used no credentials, account, Binance,
  persistence, scorer invocation, or trading authority; they are runtime
  evidence only. No prospective record, authenticated qualification,
  promotion, or Round 21 CLI assembly exists.
- `polymarket-live --action autonomous` assembles the Round 16 predictor,
  independent public predictor feed, Polymarket user stream, reconciliation,
  durable single-writer/Stop control, and settlement loops only after exact
  promotion and envelope pins. Stop is persisted before credential access and
  serialized against final order dispatch. No qualifying evidence or
  live-authority promotion exists, so the action is implemented but currently
  fails closed before opening exposure.
- Captures `eae374e2662c440fb93970d5710937b1`,
  `3a67757c7f174df4b62f2722ea9211cb`, and
  `b8a270da20fe4116a01a4626607e42da` are permanently development-only. The
  first two queues saturated. The third was terminalized `failed` after
  9,887,714 persisted messages when its indexed v2 writer reproduced the same
  long-duration throughput collapse. Storage-v3 capture
  `79ac19539d384352b865c21cb0c43627` is also permanently development-only: its
  queue reached `500000/500000` after 10.1 hours, it was deliberately stopped,
  fully drained, and terminalized `failed`. Never use any of these runs for
  model, confirmation, or profitability claims.
- Recorder storage v4 writes only bounded, checksummed frames containing exact
  payload bytes and receipt metadata; normalized events are reconstructed and
  the terminal report binds the ordered chunk-manifest root. Its 2,000,000-
  message infrastructure benchmark sustained 48,189 messages/s, replayed at
  66,049 messages/s, passed the full audit, and used 198,717,440 bytes. The
  repeated payload sample was real and hash-verified, but receipt metadata was
  synthetic and the source run failed, so this is not live-capture, model, or
  profitability evidence. A subsequent five-minute real-feed soak captured
  470,422 messages with queue high-water `569/500000`, zero recorder or
  integrity errors, and exact reopen verification. One audited CLOB disconnect
  made it `degraded`; it validates writer liveness only. The later Round 9
  confirmation supplied the required duration and synchronized-group breadth,
  but its model admission failed on unproven post-submission entry states.
- For a finished segmented Round 9 run, invoke `polymarket-action-value
  --allow-segmented-gaps` directly after official resolution. It performs and
  persists the label-free continuity audit before materialization, then reuses
  the same store's terminal integrity cache. Run standalone
  `polymarket-continuity` only when an audit-only report is needed; running both
  commands needlessly rereads the full evidence corpus.
- Round 9 MLP report v3 requires positive validation stress-utility uplift over
  ridge and at least 30 untouched synchronized test groups before reading its
  test partition. Do not weaken or bypass either admission gate.
- Round 9 maps `itode` to the independent 250 ms crypto taker delay and rejects
  nonzero general `sd`. V2 platform fees use `fd`; recorded base-fee fields are
  not additive and no builder code is modeled. The primary-source audit binds
  the official status record and SDK revisions.
- Round 9's one-second two-leg replay proves only causal CLOB book matches. It
  does not prove onchain confirmation or that newly bought tokens are sellable;
  official SELL prerequisites require confirmed conditional-token inventory.
  Keep all Round 9 outputs research-only until a separate settlement/inventory
  contract passes current source-bound failure and mark-to-market stress-test
  acceptance criteria.
- Ridge admission fails on any unproven post-submission entry state. Never
  censor or relabel it as no-fill; only a definite entry rejection such as an
  invalidated tick is a classifier-eligible zero-utility no-fill.
- Run Round 9 fits only through `polymarket-ridge` and `polymarket-mlp`. Both
  read only opaque row identities before writing a durable claim; clear labels
  load afterward. Completed claims load the signed report, while interrupted
  or failed claims block silent retries.
- Finance-LLM v6 is revoked for case-ID label leakage. V7 recorded Qwen3 8B at
  `9/11` and three 8B/9B models at `8/11`, but its permissive response parser
  invalidates the valid-JSON admission contract; keep those results as rejected
  historical evidence only. V8 preserves the 11 label-free cases and requires
  exact typed JSON. No AI model is selected. Kronos also failed its causal
  random-walk benchmark. Any AI treatment must pass current governance, then
  beat same-period non-AI execution after costs without worsening tail risk.
- AI permission maps are default-deny. Only a valid, timely approval for the
  exact hash-bound condition may permit that proposal; missing cases, malformed
  types, duplicate JSON keys, contradictory action/reason codes, low confidence,
  and latency failures remain vetoes. Single-GPU inference queue delay is
  monotonic, hash-bound, and included in effective execution latency.
- Overall AI uplift now requires both the paired primary-latency gate and
  positive, ML-beating, return- and drawdown-nondegrading execution at every
  preregistered network-latency stress. A primary-only improvement is rejected.
- The separate Round 74 event-model AI experiment must never scale or reuse the
  baseline fill. Every positive AI exposure requires an exact delayed top-20 L2
  rewalk with the AI size applied before quantity quantization. The same-entry
  latency flag is diagnostic only; accepted reviews beyond that budget retain
  their audited decision and replay the delayed book up to the frozen 30-second
  historical ceiling. Runtime vetoes, AI vetoes, expired reviews, target
  ineligibility, and delayed overlap remain paired zero-exposure outcomes.
  Baseline labels and AI replay both normalize exact walked entry notional to
  the same reference capital; requested size is never substituted for realized
  deployed notional after quantity quantization or delayed price movement.
  The first 65-minute cohort halted on slot 000 before admission because
  terminal analysis exceeded its operator budget. The later v5-r1 campaign
  halted before run creation on a transient public-WebSocket close with no
  database/WAL change. V5-r2 then created run
  `4318bfbf344a44f9987ab783b0e5489a` and captured 3,434,423 messages over
  32 minutes 40 seconds before the public source exceeded its 15-second receive
  deadline. A fresh process audited all 461 exact-wire frames, but the run and
  campaign failed closed. Queue use peaked at 15.50%, so local saturation is
  not supported; no host-wide or provider-side cause is claimed. All three
  campaigns, their slots, and their host tasks are permanently closed. Never
  retry, replace, reactivate, pool, salvage, or use any failed prefix for
  targets, models, or financial claims. Read the v5-r2 slot-000 stream-stall
  artifact for the immutable hashes and bounded diagnosis. There is no open
  Round 74 cohort capture plan. A new plan must first freeze independently
  audited transport units, explicit gap rejection, whole-unit role assignment,
  purge/embargo, and sealed-test controls without modifying historical
  evidence. Two later 20-minute prerequisite attempts also failed closed:
  attempt 001 timed out during the opening handshake before run creation, while
  attempt 002 ended on a public-WebSocket transport close after 209.781 seconds.
  Attempt 002 preserved 91,227 exact messages and passed a fresh full-frame
  audit. The v2 segmented contract now independently replays each connection
  epoch and measures usable time from feature readiness through the earliest
  final fresh BTC/ETH/SOL depth receipt. On attempt 002 that interval is only
  209.134 seconds, shorter than both the 600-second admission minimum and the
  310.5-second complete target span, so it remains permanently excluded with
  zero eligible model time. The prospective implementation uses 720
  predeclared slots and exact role-level eligible-time quotas; a qualifying
  transport-ended prefix must be included, but no feature or target may cross
  an epoch boundary. A separate attempt 003 then completed the full 20-minute
  prerequisite with 543,530 exact messages, zero reconnects, 3.05% peak queue
  use, and no WAL left behind. Independent exact-wire and causal-depth replay
  produced 536,739 tokens and 1,200.071 seconds of conservative usable time,
  so both the 600-second minimum and complete target-tail gates passed. This
  validates the capture, epoch-audit, and deterministic-adjudication mechanics,
  but the prerequisite is not cohort data. Segmented plan v2 was disabled
  before its first slot after a pre-start review found that its partition
  charged the full role embargo inside the next epoch without counting elapsed
  inter-epoch wall time. Its task never ran, and neither a v2 database nor state
  directory exists. Never reactivate or reuse v2. Gap-aware segmented plan v3
  is frozen at SHA-256
  `6c0e67a1f61247308f43112a2adcef169055abd099eb3c17a544b29372ea6c60`.
  It starts at 2026-07-28 08:00 UTC and predeclares 720 slots at 25-minute
  cadence. Windows task `SimpleAITrading-Round74-Segmented-v3` is registered
  for that exact schedule with no retry, no slot shifting, bounded resources,
  durable heartbeats, independent post-capture audit, and deterministic
  adjudication. Slot 000 ran at its exact 2026-07-28 08:00 UTC boundary and
  ended after the primary public WebSocket exceeded its receive deadline. Its
  23,833-message prefix passed exact-wire and causal-depth audit, but only
  19.5823092 seconds remained usable after feature readiness. It is therefore
  immutably `transport_excluded` with zero eligible model time and no binding.
  Never retry, pool, salvage, relabel, or use slot 000 for features, targets,
  fitting, calibration, policy selection, test evaluation, or financial
  claims. Slot 001 then completed its independent 20-minute transport epoch
  with 506,946 exact public messages, 290 frames, zero reconnects, 3.54% peak
  queue use, and no WAL left behind. Exact-wire and causal-depth replay
  produced 505,930 tokens, 1,200.140 seconds of usable time, and 889.640
  seconds of eligible anchor time. It is therefore admitted with binding
  SHA-256
  `d6deb58d3ed3e6e536aa5d89c29b0aa5a4ce2b34dae48523a2a7ede576f23752`
  as the first training run. This is not a complete cohort, model fit,
  backtest, AI comparison, or financial result. The campaign remains open;
  later scheduled slots are independent epochs. Read
  `round-074-segmented-prerequisite-attempt-003-success-2026-07-28.json` and
  `round-074-segmented-cohort-host-schedule-v2-2026-07-28.json`, then
  `round-074-segmented-v3-slot-000-transport-exclusion-2026-07-28.json`, then
  `round-074-segmented-v3-slot-001-admission-2026-07-28.json`, then
  `round-074-segmented-transport-epoch-redesign-validation-2026-07-28.json`,
  then read
  `round-074-event-sequence-model-design-v64.json`, which composes the complete
  unchanged v63 model contract with the corrected plan, operator,
  execution-evidence, and research bindings, then
  `round-074-event-sequence-model-design-v65.json`. V65 adds the only guarded
  development runner: it requires all 168 run bindings and exactly 144
  development target assemblies before opening the database, reads DuckDB only,
  rechecks capture-process and WAL absence, emits bounded-silence heartbeats,
  and never reads a sealed-test assembly. The complete cohort and target panel
  do not exist yet, so no representative training or financial result exists.
  Then read `round-074-event-sequence-model-design-v66.json`. Execution source
  schema v2 makes the venue environment intrinsic to every calibration leg and
  rejects any aggregate whose requested environment differs. Testnet execution
  records cannot be relabelled as mainnet. No testnet-to-mainnet transfer model
  exists, so mainnet target assembly and every downstream financial claim remain
  blocked. Then read
  `round-074-local-ai-review-design-v49.json`. The model now uses an exact,
  target-blind 768-window sample per capture run, disjoint 12/6/6 tuning roles,
  and bounded equal-run DirectML groups. It also adds causal
  continuous-time 5/30/300-second return, realized-volatility, order-flow,
  spread, and imbalance state to each exact event token; this corrects the
  mismatch between a 128-event window that may span only seconds and 30/300-
  second payoff targets without assuming a fixed event rate. Slow state resets
  at the five-second continuity boundary and never interpolates spread,
  imbalance, or returns through missing market time. Their source and evidence file
  hashes canonicalize text line endings to LF, so Windows and Linux verify the
  same committed content. The sealed entry point reserves
  a metadata-only identity before loading target-bearing test batches, then
  invokes the concrete two-model target-free review adapter and read-only,
  per-run exact-L2 replay adapter only while the reservation is live. Both
  adapters reconcile the full claim, model, partition, and output identities.
  The post-cohort model operator fits scaling from each unique training event,
  replays one read-only in-memory batch per capture run, and never writes an
  overlapping feature/target cache or reads the test role during development.
  A hash-bound synthetic host preflight has now run both Fino1 8B and Qwen3 8B
  through the real isolated Ollama path with 100% reported GPU residency and
  then unloaded both models. This proves local protocol, provenance, latency,
  and residency mechanics only; it is not representative-market AI evaluation,
  uplift, edge, profitability, or trading authority.
  The fixed 768-window/equal-run mechanics are superseded for the segmented
  cohort by `round-074-event-sequence-model-design-v76.json`. V76 scales each
  symbol's target-blind window quota by audited eligible wall time, selects one
  endpoint nearest each equal-time stratum midpoint, reaudits and replays each
  epoch in isolation, and gives each eligible target one gradient weight. It
  never cycles a short epoch to match a long epoch. The CPU end-to-end
  train/seal/reload test and 32-test affected checkpoint passed; DirectML
  behavior then passed separately on the host RX 9070 XT with zero fallback
  warnings; read `round-074-event-sequence-model-design-v77.json`. Real cohort
  training, P&L, drawdown, edge, and AI uplift remain untested. Do not restore
  the equal-run trainer for segmented data.
- Live Binance AI reviews are exact-case, asynchronous, hash-chained, and
  shadow-only. Pending, failed, or stale reviews block only new entries; exits
  retain the original ML side. `ai-uplift` rejects post-entry/reused reviews and
  requires contiguous one-second low/high paths from a read-only `--market-db`
  before any drawdown-preservation result can pass.
- Active per-entry AI startup also requires enough nominal candle time to
  submit and revisit the exact case under the configured poll and provider
  deadline. Impossible cadences fail before exchange setup instead of spending
  model tokens on reviews that cannot cross the entry boundary. A slower
  reusable AI risk supervisor is not implemented or implied by this gate.
- AI review v4 also requires post-inference Ollama `/api/ps` evidence for the
  exact weight digest with at least 99% of Ollama's reported model bytes in
  VRAM. Partial CPU/GPU offload is blocked. DirectML selection is separate and
  does not prove that the review model ran on GPU.
- Required-GPU AI preflight blocks unknown VRAM. Legacy ROCm output must expose
  exact total/used byte pairs; Windows AMD uses a deduplicated 64-bit driver
  total minus WDDM dedicated usage and rejects conflicting totals. The
  2026-07-16 host audit measured about 12.15 GiB free on the AMD DirectML host
  while Ollama remained unloaded. The same host audit found Ollama `0.31.2`, a
  Vulkan `1.4.349` discrete RX 9070 XT device, and no loaded Ollama model; those
  are capacity/backend facts, not inference residency or edge evidence.
- Enabled Polymarket AI also passes the shared local-GPU preflight immediately
  before provider inference. `polymarket-model` accepts mutually exclusive
  `--enable-ai` and `--disable-ai` overrides and otherwise inherits the saved
  runtime setting. The native toggle emits and smoke-tests both explicit states.
- Polymarket AI report v6 also requires exact-model terminal Ollama telemetry and
  independently reconstructs prompt/output token totals. Missing, malformed, or
  rehashed usage evidence vetoes the response; token counts do not prove edge.
- Qwen3 14B v9 is consumed. Ollama rejected all 11 requests before sampling
  because its generated schema grammar exceeded the provider's complexity
  limit. The durable claim has no score or pass state; the hash-bound incident is
  `docs/ai/risk-review/qwen3-14b-v9-infrastructure-failure.json`. Never rerun or
  reinterpret v9 as a reasoning result.
- Qwen3 14B v10 is the separately frozen one-shot governance candidate in
  `docs/ai/risk-review/qwen3-14b-v10-preregistration.json`. Its semantic cases
  are unchanged; only the provider-compatible JSON transport and failure
  evidence changed. The CLI still requires the exact preregistration, valid
  storage-v4 confirmation run, continuity evidence, unchanged model digest,
  terminal `stop`, full GPU residency, and positive coherent telemetry for all
  11 cases. Provider failures create a hash-bound failure sidecar and cannot
  produce a score. Failed claims cannot reopen cases.
- The confirmation breadth condition is evidenced by run
  `e34d349771da4c35bcc8ae436c2fe9f6`. The earlier frozen Ridge admission failure
  still blocks any claim of market edge, AI uplift, or trading authority; it does
  not turn the standalone governance benchmark into an economic experiment.
- The exact terminal facts for failed confirmation capture
  `79ac19539d384352b865c21cb0c43627` are in
  `docs/model-research/polymarket/round-009-confirmation5-failure-2026-07-16.json`.
  Its terminal integrity audit is incomplete; retain it only for recorder
  diagnosis and audit any payload sample before reuse.
- Completed confirmation capture `e34d349771da4c35bcc8ae436c2fe9f6` owns
  `data/polymarket-round9-confirmation-v4-20260716-152838Z.duckdb`. The full run
  contains audited reconnect gaps and is not globally continuous; the frozen
  label-free audit admitted only 47 independently gap-free, segment-pure groups.
  All 47 groups are replay-certified and action-materialized under implementation
  digest `5e75c49312431c3bc33c3ace33f2edf061acd6d4e6fa5c0151c76779e9f528ab`.
- Build current AI provenance with `tools/build_ai_model_provenance.py`; it must
  match protected inference evidence to the local manifest, `/api/show`
  metadata, and every blob. Never hand-edit the result or infer identity from a
  mutable tag.

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
