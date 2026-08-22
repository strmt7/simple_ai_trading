# Continue Development

This is the authoritative handoff for a new development session. Read
`AGENTS.md` and `docs/AGENT_START.md` first, then verify every stateful claim in
the repository and on GitHub before changing code.

## Current Truth

- The repository is beta `0.1.0-beta.1`. No model has production trading
  authority, and no reproducible after-cost edge currently supports a
  profitability claim.
- Development happens only on `main`. Do not create a long-lived branch or
  discard unintegrated work from another worktree.
- Binance scope is BTC, ETH, and SOL in paper or testnet/Demo mode.
- Polymarket is an independent BTC 5-minute/15-minute track. Its live-capable
  boundary is disabled by default and has no capital authority.
- Historical research has one exclusive repository cutoff:
  `2026-08-14T00:00:00Z`. Do not move it or refetch the latest history during
  each iteration. The machine contract is
  `docs/model-research/research-data-snapshot-contract-v1.json`.
- Prospective captures after the cutoff are isolated, append-only experiments.
  They must never be merged into the frozen historical snapshot or used after
  their preregistered access boundary is violated.
- The old Round 21/27 PIDs (`33804`, `34228`, `65092`, `36844`) are no longer
  alive. Do not restart them from this historical note.

## Live Host Handoff: 2026-08-22

The following state was reverified through approximately
`2026-08-22T17:14Z`. It is host state, not model evidence, and must be
rechecked before any action.

- Main and `origin/main` were synchronized at model-evidence refactor revision
  `9f5efe1573281306a1dde413afe353fcc309bee3` before this documentation
  closeout. GitHub's remote branch inventory contained only `main`, with no
  open pull requests. Always verify
  `git rev-parse HEAD`, `git rev-parse origin/main`, and the remote branch
  inventory directly before resuming.
- Round 75 prospective Binance capture is active from the detached worktree
  `C:\trader\simple_ai_trading-model-dev`. The latest supervisor record was
  `service_healthy`, with no credentials, orders, or trading authority.
  Scheduled task `SimpleAITrading-Round75-Continuous-Capture-Supervisor-v1`
  was ready and its `2026-08-22T17:13:15Z` invocation returned `0`. The service
  state was fail-closed at slot `674`, phase `waiting_for_next_fixed_slot`.
  The state tree contained 36 completed slot directories and 639 immutable
  missed-slot receipts, all classified as
  `service_observed_start_window_elapsed_without_reservation`, with exact host
  cause not established and automatic retry forbidden. The service stderr log
  also contained repeated `Round 75 storage resource gate failed` records.
  Do not infer causality between those facts, salvage missed slots, or modify
  the frozen process; perform the contract-defined terminal resource and
  continuity audit after the campaign ends at `2026-08-23T12:00:00Z`.
- The independent Round 21 Binance spot/futures sidecar is active from clean,
  detached commit `4a5912574a9157d79fecc53bf68ed6f01bb8dac8` in
  `C:\trader\simple_ai_trading-round21-sidecar-v2`. Wrapper PID `35008` and
  physical child PID `35264` were alive. Its state reported segment `12`,
  phase `capturing`, `123162645` written public messages, twelve recorded
  stream gaps, zero recorder errors, no credentials, no execution connection,
  no model-data eligibility, and no paper or live authority. Its frozen
  scheduled end is `2026-08-29T23:40:00Z`.
- All process IDs above are ephemeral snapshots. Process rotation is expected;
  never infer failure, completion, or permission to restart from PID drift.
- Never stop, restart, stage, clean, reset, switch, merge, or modify either
  active capture worktree from this handoff. PID death alone is not a terminal
  verdict; use the contract-defined state, lease, process ancestry, scheduled
  task, database/WAL, and independent terminal audit together.

### Unintegrated Worktree

The Round 75 worktree is detached at
`c42219d47dc781a46411a4ec96838f8a26c3924c`, which is an ancestor of current
`main`. Relative to that old base, Git reports 99 tracked changes and 218
untracked paths. That status is not an unpublished-work count: comparison with
published `main` at `49195b214cdbfd8a188bafa71426e4e7889478c2` found 218
byte-identical files, two local deletions already absent from `main`, 71 files
with different content, and 26 local files absent from `main`. Therefore 97
local content paths require post-capture review; the other 220 must not be
needlessly copied or recommitted. At the live snapshot its tracked binary diff
hashed to Git object
`bf7f896c3fa2b17a7a7a34887b2d3fe04cb4be54`; the sorted untracked
path/content manifest had SHA-256
`f622661d58b6cf044d9e0946c7115ceffa379de553f2eaecaee5b999b02be13b`.
The newest untracked-file write was `2026-08-10T21:43:27Z`.

These hashes are preservation aids, not validation or approval. Do not commit
or change HEAD in that worktree while Round 75 is live: the capture is bound to
its frozen implementation and a repository identity change could invalidate
it. After contract-defined terminalization, snapshot the state again, preserve
the complete diff and untracked manifest, and reconcile it onto `main` with a
reviewed three-way integration. Never use `git clean`, destructive reset, or a
blind overwrite. All new development after integration belongs on `main`.

### Security Snapshot

- GitHub reported zero open Dependabot alerts, zero open secret-scanning
  alerts, no open pull requests, and only `main`. GitHub code scanning returned
  `404: no analysis found`; it is unavailable, not zero.
- `uv lock --check` passed. Dependency manifests and the lock have not changed
  since pip-audit found no known vulnerabilities in the complete exported GPU
  and DirectML dependency stacks. The exact pushed revision
  `053e71ec9959c7b51c73d6983fb574e03f4173ea` passed hosted actionlint and
  Zizmor `1.25.2` through Super-Linter.
- A fresh Bandit audit of `src/simple_ai_trading` reported 522 audit findings:
  zero high severity, 251 medium, and 271 low.
  The medium set is dominated by 248 dynamic-SQL (`B608`) review items. Its
  three `B113` request findings are false positives because each displayed call
  supplies an explicit bounded timeout. The remaining findings still require
  source-by-source triage; this audit does not prove vulnerability or safety.
- `market_store._table_columns` was source-traced: its only caller supplies the
  fixed internal literal `archive_files`, so that `B608` item has no
  attacker-controlled source-to-sink path and needs no speculative patch.
- A Windows child-process reproducer proved that a fake `git.exe` in the
  process current directory could satisfy an unqualified `git` invocation.
  Package bootstrap now removes empty, relative, duplicate, and
  current-directory PATH entries before any operational module runs and sets
  `NoDefaultCurrentDirectoryInExePath=1` on Windows. The paired exploit test
  proves the unguarded child executes the fake binary while the guarded child
  does not. The 22 `B607` calls remain scanner-visible because Bandit cannot
  model package bootstrap; they are not suppressed. Explicit absolute PATH
  directories remain a host trust boundary and still require controlled
  installation and permissions.
- Two silent terminal-holdout audit exceptions were removed. If terminal
  evaluation or result fingerprinting fails, the model remains rejected; a
  successful ledger update records `evaluation_error`, while a failed update
  leaves the durable reservation in fail-closed `reserved` state and now adds
  `terminal_audit_error` to rejection diagnostics without replacing the
  primary failure. The affected 96 training-suite and terminal-ledger tests,
  Ruff, and a focused Bandit scan pass.
- The remaining two `B110` and one `B112` findings were then removed without
  suppressions. Per-device CUDA memory-probe failures are debug-observable and
  do not hide usable peer devices. A failed Polymarket recovery rollback is
  attached to the primary exception with unknown transaction state. Round 74
  testnet transport cleanup attempts WebSocket, listen-key, and HTTP-session
  closure, reports every failure, and does not mask an active primary error.
  Focused tests, mypy, Ruff, and the full `B110`/`B112` scan pass. These changes
  do not alter model outputs, evidence, risk limits, or trading authority.
- AI uplift schema `ai-uplift-v5` now fails closed when aggregate ROI does not
  strictly improve over the machine-learning baseline. It records the exact
  metric alias used by each treatment, rejects baseline/AI alias mismatches,
  normalizes explicit drawdown and win-rate percentage aliases, validates
  count/range inputs, and requires strict aggregate P&L, ROI, and expectancy
  improvement without worse maximum drawdown. The independent financial-sanity
  reader validates v5 provenance, recomputes every reported delta, and applies
  the same directional gates so a hand-edited `accepted` flag cannot bypass the
  production assessor. This is evidence-contract hardening, not AI uplift,
  market-edge, or profitability evidence.
- Code checkpoint `5c220bcfa34028640fb3ac00d8953314c291d7fd`
  passed 189 focused AI/uplift/review/financial-sanity/Polymarket tests, local
  Ruff check, Pylint errors-only, targeted Bandit, terminology, lock, and diff
  checks. Hosted Vulture and Super-Linter passed. Hosted Ruff failed because a
  newly added long expression did not satisfy its changed-range format gate;
  the closeout formats only the new ranges and locally reproduces zero overlap.
  DeepSource failed with 32 displayed issues, 34 resolved, and 13 classified as
  introduced. The blocking items were strict object-to-number type errors in
  the newly reclassified `ai_uplift.py`; the remainder shown were legacy
  complexity findings. The closeout adds typed period/bootstrap contracts and
  guarded finite/integer/reason parsing. Current mypy then reports zero issues
  in both changed source modules, compared with 18 on exact parent `486f0506`.
  The local closeout passes 191 affected tests and 96% changed-line coverage,
  plus Ruff check, Pylint errors-only, targeted Bandit, markdownlint,
  terminology, lock, and diff checks. Hosted Ruff, Vulture, Super-Linter, and
  the Windows Python/native-UI job then passed on closeout revision
  `f50c3a8302090b15bce8075f425716dcf2afc659`. Its full Linux
  test-and-coverage job was still running at the handoff and is not a pass
  until GitHub reports a successful terminal result. DeepSource reanalysis
  resolved the displayed type-contract findings. It still failed with only
  `PY-R1000` cyclomatic-complexity findings visible in the changed Python
  scope: four in `ai_uplift.py` and five in `financial_sanity.py`. These are
  maintainability findings, not reported security vulnerabilities; preserve
  behavior and reason ordering if addressing them in a later bounded refactor.
- Full Linux CI on documentation closeout revision `68c1a422` exposed 29
  cascading Round 27/28 failures. The immutable v7 source ledger and cumulative
  v17 replacement layer no longer recognized the later `b5144c98`
  exception-path changes in `compute.py` and `polymarket_recorder.py`. The
  cumulative v18 maintenance amendment preserves both predecessors and binds
  the original ledger hashes to the current implementations. It changes no
  model mathematics, feature, target, cost, threshold, risk gate, authority, or
  result. The exact previously failing cluster now passes 70 tests locally with
  100% changed-line coverage, plus Mypy, Ruff/format, Pylint errors-only,
  targeted Bandit, lock, manifest, and diff checks. Repair revision `580e4a0e`
  subsequently passed full hosted CI, Ruff, Vulture, Super-Linter, DeepSource,
  and the Windows Python/native-UI smoke job.
- The four `ai_uplift.py` complexity findings from `f50c3a83` were decomposed
  in revision `9f5efe15` without changing output construction or rejection
  order:
  `AIUpliftPolicy.__post_init__` fell from 24 to 1, `_matched_period_deltas`
  from 22 to 9, `_statistical_evidence` from 20 to 2, and `assess_ai_uplift`
  from 33 to 3. The affected matrix passes 217 tests and the 156 changed
  executable lines have 100% coverage. Mypy, Ruff/format, Pylint errors-only,
  targeted Bandit, and Radon passed locally. Exact-SHA hosted CI, Ruff,
  Vulture, Super-Linter, DeepSource, and the Windows Python/native-UI smoke job
  also passed.
- The five deferred `financial_sanity.py` paths were subsequently decomposed
  without changing calculations, check order, labels, paths, metrics, limits, or
  report serialization. Radon scores fell from `188` to `2` for
  `build_model_lab_financial_sanity_report`, `125` to `2` for
  `build_backtest_financial_sanity_report`, `64` to `5` for
  `_selection_risk_checks`, `36` to `1` for
  `_ai_uplift_period_evidence_checks`, and `30` to `1` for
  `_probability_calibration_checks`. The module's highest remaining score is
  `23`. Four fixed SHA-256 fingerprints cover complete good/bad backtest and
  model-lab reports. Locally, 49 focused financial-sanity tests and the 233-test
  affected matrix pass; changed executable-line coverage is `95.7%` (`741`
  lines, `32` missing), clearing the `95%` gate. Ruff/format, Pylint
  errors-only, targeted Bandit, and Radon pass. Mypy reports no error in the
  changed module but still follows imports into three unchanged modules with
  nine recorded type errors. At that local checkpoint, hosted results were
  unverified; the next bullet records the first exact-SHA analyzer result.
- DeepSource failed revision `eaff89305fca00073e8d701bd8a85ad0acc0d920`
  with 15 changed-scope maintainability findings and no displayed security
  finding: 13 helper functions exceeded its strict complexity threshold, one
  nested conditional was collapsible, and one comparison could be chained.
  The two simplification findings are fixed in the immediate follow-up. The
  residual complexity queue is `_walk_forward_gate_checks` (`23`),
  `_terminal_holdout_check` (`23`), `_ai_uplift_count_checks` (`22`),
  `_ai_uplift_period_contract_checks` (`19`), `_market_edge_checks` (`19`),
  `_ai_uplift_rate_checks` (`19`), `_ai_uplift_bootstrap_checks` (`18`),
  `_backtest_sequence_checks` (`18`), `_backtest_log_identity_checks` (`18`),
  `_probability_deterioration_checks` (`17`),
  `build_model_financial_sanity_report` (`17`), `_data_coverage_checks` (`16`),
  and `_accepted_portfolio_symbol_checks` (`16`). Do not broaden this batch or
  suppress the rule during closeout. A later agent should decompose one coherent
  contract family at a time under the fixed full-report fingerprints and then
  verify the exact new DeepSource inventory.
- The next local model-evidence batch reduces four functions from that exact
  queue without changing calculations or diagnostics:
  `_ai_uplift_period_contract_checks` falls from `19` to `8`,
  `_ai_uplift_bootstrap_checks` from `18` to `9`,
  `_probability_deterioration_checks` from `17` to `8`, and
  `build_model_financial_sanity_report` from `17` to `1`. The largest extracted
  predicate scores `12`, below DeepSource's observed threshold. Accepted and
  rejected model reports add two fixed SHA-256 fingerprints to the four existing
  complete report contracts. Locally, 49 focused tests and the 233-test affected
  matrix pass, and all `62` changed executable lines have coverage. Scoped Mypy,
  Ruff/format, Pylint errors-only, targeted Bandit, strict simplification checks,
  and Radon pass. This is maintainability evidence only; hosted DeepSource must
  confirm the residual inventory on the exact pushed SHA.
- Exact revision `67dab5d8d4acbb12d71077a2fa30cd62ac5f03f3` subsequently
  passed CI, Ruff, Vulture, Super-Linter, and Windows native-UI smoke.
  DeepSource confirmed that the four targeted findings were gone and reported
  exactly the expected nine residual changed-scope complexity findings.
- The current local batch decomposes all nine residual paths. Their Radon scores
  are now `5/3/6/14/14/12/14/1/1` in the documented DeepSource order, and no
  replacement helper exceeds `14`. The six complete accepted/rejected report
  fingerprints remain unchanged. Locally, 51 focused tests and the 235-test
  affected matrix pass, all 133 changed executable lines have coverage, and
  scoped Mypy, Ruff/format, Pylint errors-only, targeted Bandit, strict
  simplification checks, and Radon pass. Hosted DeepSource must analyze the exact
  pushed checkpoint before this changed-scope queue is called closed.
- Exact parent `486f0506d60857e291941c7f17580c172b8ea5ca`
  passed hosted Ruff, Vulture, Super-Linter, and DeepSource. Its longer CI run
  was still in progress at the snapshot.
- The last full DeepSource backlog verification on `2026-08-16` found about
  28000 active findings, including 118 security findings. DeepSource passed
  revision `053e71ec9959c7b51c73d6983fb574e03f4173ea` with no blocking issues or
  failing metrics; the historical project-backlog count was not refreshed and
  must not be represented as current.
- DeepSource failed exact revision
  `c6ff3ac286eca7d58938ff9eded3c461f9ad60ea` with 107 Python findings, 88
  classified as introduced. Its first rendered page contained fifteen major
  type-contract findings in `backtest.py`, `execution_simulation.py`, and
  `financial_sanity.py`; it did not identify those findings as security
  vulnerabilities. Revision
  `16c40e26d278caf5eba166609fb70716325adaf1` added explicit
  numeric-conversion guards, a typed scoring-backend payload, unambiguous
  optional trade-limit and threshold-variant types, and primitive-only sanity
  diagnostics. Its hosted DeepSource run reduced the exact inventory to 86
  findings, reported 536 resolved findings, and classified 75 remaining
  findings as introduced. The follow-up closeout revision removes the next
  page's loop-variable closure captures, one no-op assignment, one unnecessary
  generator, and a simple comparison warning. It also makes two accelerated
  scorer parity tests execute through Torch on CPU when DirectML is absent.
  Focused behavioral tests, mypy, Ruff, Bandit, and Pylint pass locally. Large
  function-complexity findings and the rest of the DeepSource backlog remain.
  DeepSource failed follow-up revision
  `2457d9d59f2c680998d2e296a0d4ce5639c5f049`: its headline Python count fell
  to 51, while the change summary reported 61 issues, 10 resolved and 51
  introduced. Its first rendered item was the intentionally deferred
  cyclomatic-complexity finding for `_batch_probabilities_torch`; do not infer
  that the remaining backlog is fixed or that every finding is a vulnerability.

### Round Closeout Checkpoint

Code revision `5c220bcfa34028640fb3ac00d8953314c291d7fd` and closeout revision
`f50c3a8302090b15bce8075f425716dcf2afc659` were pushed to `main` with the
AI-uplift v5 contract described above. At the snapshot, GitHub showed only
`main`, no open pull requests, zero open Dependabot alerts, and zero open
secret-scanning alerts. GitHub code scanning returned `404: no analysis found`
and remained unavailable. Hosted Ruff, Vulture, Super-Linter, DeepSource, and
the Windows job passed on documentation closeout `68c1a422`; full Linux CI
failed only the 29 source-provenance tests described above. Cumulative v18
repair `580e4a0e` then passed every hosted workflow and DeepSource. The four
`ai_uplift.py` findings were closed by `9f5efe15`, which passed every available
hosted workflow and DeepSource. The five `financial_sanity.py` findings were
then closed locally under full-report fingerprint tests as described above;
their hosted status must be read from the exact final checkpoint. GitHub still
reported zero open Dependabot and
secret-scanning alerts; code scanning remained unavailable with
`404: no analysis found` and is not a zero-alert result.
No model score, label, fill, cost, threshold, model outcome, trading authority,
or profitability claim changed. Verify every available hosted check on the
final pushed SHA; pending and unavailable checks are not passes.

This is an interruption-safe checkpoint, not product completion. Round 75 and
Round 21 remain active under their frozen contracts. The 97 modified, 2 deleted,
218 untracked, and 71 differing plus 26 locally-only path counts above are a
historical preservation snapshot against an older `main`; they must be
recomputed after terminalization because `main` has advanced. Do not reconcile
any path before Round 75 terminalization.

## Completed Foundation

This section describes pushed `main` only. It predates the unintegrated
Round 74/75 worktree above and must not be used to infer that newer local work
is published, tested on current `main`, or safe to discard.

- CLI and native Windows commands share one parser-derived contract.
- Deterministic ownership, reconciliation, Pause, Stop, loss, liquidity,
  latency, rate-limit, and stale-state gates remain outside AI control.
- Binance and Polymarket execution, capital, credentials, and order ownership
  are separate.
- Current model and AI research records preserve rejected, blocked, and
  non-authoritative outcomes instead of relabeling them as success.
- Round 29 now has a hash-bound, target-blind six-field settlement interaction
  overlay and preregistration. It repairs a design/implementation gap for the
  linear residual candidate without changing frozen Round 27/28 rows or
  pretending to reproduce Chainlink's unpublished TWAP method. Synthetic
  transform, zero-variance, composition, and tamper tests pass; no Stage 1 row,
  outcome, model metric, P&L, or authority was produced.
- Dependencies, imported agent workflows, release automation, and concise
  operator documentation are integrated into the main-line closeout gate.
- The 2026-08-16 DeepSource remediation addresses the complete 219-finding
  inventory across 25 Python modules without analyzer suppressions. Ruff,
  format, Vulture, terminology, provenance, badge, and manifest checks pass
  locally. The final full pytest run passed at 100%; source branch coverage is
  76.42%, and changed-line coverage is 95% (765 executable lines, 38 missing).
  Native Windows build, CLI parity, smoke, launcher, and layout checks also
  pass; the audited 2250x1470 render has no clipping or overlap.
- A complete Codex Security diff review found no reportable findings across all
  30 changed product-source files. Dependency audits of both supported GPU
  stacks found no known vulnerabilities. Zizmor auditor and pedantic scans pass
  without a long-lived coverage-service token. Coverage remains enforced in CI
  and published as a GitHub artifact; the unavailable, unprovisioned Codecov
  upload is not treated as a repository quality gate. Before this commit, live
  GitHub showed zero open Dependabot alerts, zero open secret-scanning alerts,
  and only `main`; code scanning was unavailable and remains unverified. Hosted
  CI, Ruff, Vulture, Super-Linter, and DeepSource are never inferred from local
  runs; this checkpoint is valid only if they pass on its pushed commit.
- Immutable Round 27 source-ledger and Round 28/29 preregistration hashes remain
  preserved. Canonical source-only remediation amendments bind every
  semantics-preserving analyzer refactor before any Stage 1 feature or outcome
  access. Round 28 correction v3 supersedes v2 only for those source bindings;
  it creates no data, model result, P&L, or authority. The generated publication
  manifest now lists Round 29 as the latest research round without claiming a
  result or authority.

### Latest Model Audit

A read-only audit on `2026-08-22` traced the unintegrated Round 74 v136
target-free pretraining objective through feature-view masking, target
construction, loss normalization, causal encoders, chronological purged
splits, segmented population weighting, paired random-state restoration, and
the focused failure-mode tests. The implementation matched its frozen
mask-aware contract: event targets come from untouched observations, masked
continuous dimensions are excluded from both target scoring and the loss
denominator, segmented rows are visited once without cycling, and validation
uses the same eligible-row weighting. No evidence-backed defect was found, no
training or profitability claim was made, and no source or capture worktree was
changed. Re-run the focused tests only after the capture is terminalized and
the differing files are safely reconciled onto `main`.

The same audit identified a post-capture, preregistration-only challenger: keep
v5 unchanged and compare a separate next-mark plus conditional-duration-density
objective on identical encoders, splits, seeds, populations, supervised
training, and after-cost evaluation. The duration target must come from the
untouched next event even when timing input is masked. Call it a conditional
marked-duration density, not a complete point-process likelihood, unless the
dataset exposes observation-end censoring and terminal survival. Promotion
requires paired proper-loss improvement without run/symbol/task subgroup
degradation and then downstream after-cost improvement. Do not implement or
evaluate it before Round 75 terminalization; it has no present edge claim.

## Next Work

1. Reverify both live captures and their canonical state files. Let Round 75
   reach its fixed terminal boundary and let the Round 21 sidecar reach its
   separately frozen boundary. Do not inspect targets or outcomes early.
2. Terminalize each capture only through its frozen contract. Run target-blind
   source, continuity, gap, role, population, lease, database/WAL, and resource
   audits before deciding whether any row is admissible. A failed gate closes
   that lineage; it does not permit salvage or threshold changes.
3. After Round 75 terminalization, preserve and re-hash the entire detached
   worktree before integration. Recompute its content comparison against the
   then-current `main`; the 2026-08-22 snapshot had 71 differing and 26
   locally-only files, while 220 status paths were already equivalent. Inspect
   every differing path and preserve the analyzer fixes already on `main`; do
   not bulk-copy or stage all dirty-status paths. Commit and develop only on
   `main` after that point.
4. Treat the unintegrated Round 74/75 documents and implementation as
   unverified until focused domain tests, contract hashes, generated CLI/native
   parity, and the affected analyzer gates pass on the integrated tree. Do not
   publish or infer their model claims from local file presence.
5. Continue security triage in bounded batches. The `B110` and `B112` queue is
   now zero. Trace remaining Bandit/DeepSource dynamic SQL source-to-sink paths.
   Distinguish fixed or validated identifiers and trusted executable contracts
   from actual injection or path-hijack exposure. Add direct regressions for
   every behavior change; do not suppress the backlog broadly. Continue from
   the exact DeepSource inventory on the final closeout SHA. The first
   type-contract batch reduced the headline Python count from 107 to 86 and the
   follow-up reduced it to 51. The nine changed-scope AI-uplift and
   financial-sanity complexity findings were later closed in two bounded,
   behavior-preserving batches. DeepSource then exposed the 13 smaller helper
   findings listed above under its stricter threshold. Exact revision `67dab5d8`
   closed four and confirmed the remaining nine. The subsequent local batch
   addresses all nine; continue from the exact final-SHA DeepSource inventory
   rather than assuming they are closed. Do not treat this changed-scope closure
   as a project-wide backlog or security result.
6. Evaluate a model only after source, causal split, cost, delay, access-ledger,
   sample, and implementation bindings are complete. AI remains veto/downsize
   only until matched, latency-charged causal uplift is demonstrated.
7. Run focused checks while integrating, then the full local matrix once at the
   final checkpoint. On this Windows host invoke pytest through
   `uv run --locked python -m pytest`; the direct `uv run pytest` console entry
   does not place the repository's `tools` package on `sys.path`. Push `main`,
   verify all hosted workflows and available scanners on that exact SHA, and
   keep the GitHub branch inventory to `main`.

## Safety Invariants

- Conservative remains default; reinvestment remains off.
- Leverage is a ceiling, not edge. Any gate may reduce size to zero.
- Only provably bot-owned orders and parent-bound positions may be changed.
- Unknown market, account, fill, fee, order, or redemption state blocks new
  exposure. Reconnect recovery reconciles first and observes fresh books before
  deciding.
- AI never submits orders, raises risk, overrides a safety gate, or blocks a
  close.
- Canonical numeric evidence is JSON/CSV. Charts and prose are derived views.
  Never invent, repair by hand, or silently replace market data or results.

## Evidence Map

- Binance status: `docs/model-research/action-value/latest/README.md`
- Polymarket status: `docs/model-research/polymarket/latest/README.md`
- Round 29 preregistration:
  `docs/model-research/polymarket/round-029-settlement-state-matched-ablation-preregistration-v1.json`
- Model rules: `docs/MODEL_AND_SIGNAL_VALIDATION.md`
- Data cutoff: `docs/model-research/research-data-snapshot-contract-v1.json`
- Agent and CI workflow: `docs/AGENT_WORKFLOWS.md`
- Product direction: `PLANNING.md`
- Live local capture and integration boundary: this document's
  `Live Host Handoff: 2026-08-22` section

Suggested continuation request:

> Read `AGENTS.md`, `docs/CONTINUATION.md`, and `PLANNING.md`; verify the current
> repository, active processes, and GitHub state; then continue the ordered
> work without weakening frozen evidence or safety gates.
