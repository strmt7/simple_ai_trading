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
  a live inference payload. Unknown order state blocks new exposure.
- Secrets must never enter prompts, logs, artifacts, tests, commits, or docs.

## Task routing

| Task | Read first | Canonical evidence |
|---|---|---|
| Binance model or backtest | nearest model module and test | `docs/model-research/action-value/latest/README.md`, then selected rows from `progress.csv` |
| Prior model failure | last row plus the relevant mechanism row in `docs/model-research/action-value/latest/progress.csv` | that row's named design/report only |
| Polymarket model | Round 13 program/evaluator and focused tests; use Round 11 only for predecessor diagnostics | `docs/model-research/polymarket/latest/README.md`, then the Round 13 contract |
| Polymarket recorder/replay | matching recorder or replay module and test | `docs/model-research/polymarket/prospective-continuity-contract-v2.json` |
| Risk or execution | nearest risk/execution module and test | `docs/LIVE_MARKET_SIMULATION.md` or `docs/POLYMARKET_PAPER_TRADING.md` only at the relevant heading |
| AI provider/model | nearest AI module and test | `docs/ai/risk-review/latest/comparison.json` and its sibling provenance |
| CLI | command handler, parser definition, and CLI tests | parser-generated help; do not infer parity from docs |
| Windows app | `src/simple_ai_trading/command_contract.py`, `src/simple_ai_trading/windows_app.py`, and the UI/parity tests | `native/windows/generated/command_contract.hpp` and `tests/test_ai_runtime_and_parity.py`; edit the shared taxonomy, never the generated header |
| CI/release | one workflow and its test/lint config | `docs/AGENT_WORKFLOWS.md` |
| Broad architecture | `docs/SIMILAR_TRADING_REPOS_REVIEW.md` | source and tests for each affected boundary |

The native app verifies the backend's `ui_contract` SHA-256 before any ordinary
workflow. A mismatch blocks Start and expert commands but never Pause or Stop.
After parser or taxonomy changes, regenerate the header through the native build.

## Model research state

- The compact cross-round ledger is
  `docs/model-research/action-value/latest/progress.csv`. Read its header and
  only the last few rows unless a task names an older mechanism.
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
  checks. No eligible seven-day cohort exists yet. Prospective v9 feature
  collection and selected-anchor target v2 implementation remain open; target
  results, model evaluation, and all trading authority remain closed.
  One hour is not a model-evaluation corpus.
  The segmented-corpus and rotation-runner contracts are now frozen. Historical
  runner v1 rows remain independently auditable, while current runner v2 admits
  v9 capture, reports, and recovery only. It uses one lease owner, terminal
  batch journals, zero reconnects, one-hour segments, recovery-before-capture,
  and serial exact replay after capture.
  Recovery-only batch `6d8c31559bb044b3a83fdf9e771dda4a` passed its real
  lease, discovery, terminal-journal, release, and independent audit paths with
  zero database growth. This authorizes one live runner segment. Do not start a
  multi-segment collection until that segment and its deep batch audit pass.
- Round 61 rejected elevated-funding spot/perpetual carry on capacity, median
  after-cost return, and lower-confidence-bound gates. Do not tune or retrain
  that family.
- Polymarket Round 13 is frozen but has not started. It is a one-use prospective
  24-hour confirmation of the unchanged Round 11 calibration, with label-free
  treatment/control decisions, full displayed-depth FOK simulation, explicit
  worst-price limits, seven execution stresses, and conjunctive activity,
  uncertainty, drawdown, and exposure gates. Its current V2 FOK BUY model first
  requires the live CLOB protocol version, then uses exact quote cents and walks
  share-denominated asks. The amount and signed shares both satisfy the recorded
  numeric minimum because official public material does not specify that field's
  BUY unit; signed-minimum, decision-book-modeled, and post-latency-modeled
  quantities remain separate evidence. Model selection and scored utility use
  only the signed minimum, never unobservable price-improvement shares.
  Round 12 is invalidated before
  outcome access and is neither model nor performance evidence. Round 11 remains
  the latest scored result and is rejected. It reused the real
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
