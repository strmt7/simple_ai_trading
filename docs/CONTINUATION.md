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

The following state was reverified at approximately `2026-08-22T13:09Z`. It is
host state, not model evidence, and must be rechecked before any action.

- The bounded audit began from clean, synchronized parent
  `a9eb70a13ffefcd56055a628054c1c177fdb55dc`; local handoff changes were in
  progress at this host snapshot. GitHub and local remote-tracking inventory
  contained only `main` after pruning. Do not copy a commit identifier from
  this mutable handoff; verify `git rev-parse HEAD` and
  `git rev-parse origin/main` directly.
- Round 75 prospective Binance capture is active from the detached worktree
  `C:\trader\simple_ai_trading-model-dev`. Canonical state named parent PID
  `37976` and physical child PID `2928`, reported
  `waiting_for_next_fixed_slot`, slot ordinal `665`, no credentials, no orders,
  and no trading authority. The frozen campaign ends at
  `2026-08-23T12:00:00Z`. Scheduled task
  `SimpleAITrading-Round75-Continuous-Capture-Supervisor-v1` was ready; its
  `2026-08-22T13:07:15Z` invocation succeeded.
- The independent Round 21 Binance spot/futures sidecar is active from clean,
  detached commit `4a5912574a9157d79fecc53bf68ed6f01bb8dac8` in
  `C:\trader\simple_ai_trading-round21-sidecar-v2`. Wrapper PID `35008` and
  physical child PID `35264` were alive. Its state reported segment `12`,
  phase `capturing`, `115951163` written public messages, twelve recorded
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
- A fresh Bandit audit of `src/simple_ai_trading` reported 525 audit findings:
  zero high severity, 251 medium, and 274 low.
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
  Ruff, and a focused Bandit scan pass. The full Bandit inventory now contains
  two `B110` and one `B112` findings: rollback/transport cleanup and an optional
  GPU-memory probe. They need bounded source-level review, not blanket
  suppression.
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

Revision `2457d9d59f2c680998d2e296a0d4ce5639c5f049` was pushed to `main` and its
hosted Ruff and Vulture checks passed. Its longer Python, Windows-native UI, and
Super-Linter jobs were still running at this handoff. DeepSource failed with
the reduced exact inventory described above. GitHub showed only `main`, no open
pull requests, zero open Dependabot alerts, and zero open secret-scanning
alerts; code scanning remained unavailable. The bounded follow-up changes no
model scores, labels, fills, costs, thresholds, model evidence, trading
authority, or profitability claim. Verify every available hosted check on the
final pushed SHA; a pending or unavailable check is not a pass.

This is an interruption-safe checkpoint, not product completion. Round 75 and
Round 21 remain active under their frozen contracts. The detached Round 75
worktree still reports 97 modified paths, 2 deleted paths, and 218 untracked
paths; the content review boundary remains 71 differing plus 26 locally-only
paths. Do not reconcile any of them before Round 75 terminalization.

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
5. Continue security triage in bounded batches. Review the remaining two
   `B110` cleanup paths and one `B112` optional hardware probe, then trace
   Bandit/DeepSource dynamic SQL source-to-sink paths. Distinguish fixed or
   validated identifiers and trusted executable contracts from actual
   injection or path-hijack exposure. Add direct regressions for every behavior
   change; do not suppress the backlog broadly. Continue from the exact
   DeepSource inventory on the final closeout SHA. The first type-contract
   batch reduced the headline Python count from 107 to 86 and the follow-up
   reduced it to 51; the remaining complexity findings were intentionally
   recorded rather than rushed into this closeout.
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
