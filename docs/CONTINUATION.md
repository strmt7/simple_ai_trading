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

The following state was verified at approximately `2026-08-22T07:54Z`. It is
host state, not model evidence, and must be rechecked before any action.

- Pushed `main` is clean at
  `9bd824a71bfa08f8e1d6e066f79867174d778d00`, synchronized with
  `origin/main`. GitHub and local remote-tracking inventory contain only
  `main` after pruning.
- Round 75 prospective Binance capture is active from the detached worktree
  `C:\trader\simple_ai_trading-model-dev`. Parent PID `33332` and physical
  child PID `29136` were alive. Canonical service state reported
  `waiting_for_next_fixed_slot`, slot ordinal `652`, no credentials, no orders,
  and no trading authority. The frozen campaign ends at
  `2026-08-23T12:00:00Z`. Scheduled task
  `SimpleAITrading-Round75-Continuous-Capture-Supervisor-v1` was ready and its
  most recent invocation succeeded.
- The independent Round 21 Binance spot/futures sidecar is active from clean,
  detached commit `4a5912574a9157d79fecc53bf68ed6f01bb8dac8` in
  `C:\trader\simple_ai_trading-round21-sidecar-v2`. Wrapper PID `35008` and
  physical child PID `35264` were alive. Its state reported segment `12`,
  phase `capturing`, `102506595` written public messages, ten recorded stream
  gaps, no credentials, no execution connection, no model-data eligibility,
  and no paper or live authority. Its frozen scheduled end is
  `2026-08-29T23:40:00Z`.
- Never stop, restart, stage, clean, reset, switch, merge, or modify either
  active capture worktree from this handoff. PID death alone is not a terminal
  verdict; use the contract-defined state, lease, process ancestry, scheduled
  task, database/WAL, and independent terminal audit together.

### Unintegrated Worktree

The Round 75 worktree contains substantial work that is not on GitHub. It is
detached at `c42219d47dc781a46411a4ec96838f8a26c3924c`, which is an ancestor of
current `main`, with 99 tracked changes and 218 untracked files. At the live
snapshot its tracked binary diff hashed to Git object
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
- `uv lock --check` passed. Pip-audit found no known vulnerabilities in the
  complete exported GPU and DirectML dependency stacks. Zizmor pedantic found
  no workflow findings.
- A fresh Bandit audit of `src/simple_ai_trading` scanned 436585 lines and
  reported 527 audit findings: zero high severity, 251 medium, and 276 low.
  The medium set is dominated by 248 dynamic-SQL (`B608`) review items. Its
  three `B113` request findings are false positives because each displayed call
  supplies an explicit bounded timeout. The remaining findings still require
  source-by-source triage; this audit does not prove vulnerability or safety.
- The last live DeepSource verification on `2026-08-16` passed the final commit
  delta with zero introduced issues, while its project backlog remained about
  28000 active findings, including 118 security findings. Recheck DeepSource
  before changing that claim.

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

## Next Work

1. Reverify both live captures and their canonical state files. Let Round 75
   reach its fixed terminal boundary and let the Round 21 sidecar reach its
   separately frozen boundary. Do not inspect targets or outcomes early.
2. Terminalize each capture only through its frozen contract. Run target-blind
   source, continuity, gap, role, population, lease, database/WAL, and resource
   audits before deciding whether any row is admissible. A failed gate closes
   that lineage; it does not permit salvage or threshold changes.
3. After Round 75 terminalization, preserve and re-hash the entire detached
   worktree before integration. Reconcile its 99 tracked and 218 untracked
   files onto current `main` without losing the analyzer fixes already on
   `main`. Inspect every conflict; do not bulk-copy over newer files. Commit and
   develop only on `main` after that point.
4. Treat the unintegrated Round 74/75 documents and implementation as
   unverified until focused domain tests, contract hashes, generated CLI/native
   parity, and the affected analyzer gates pass on the integrated tree. Do not
   publish or infer their model claims from local file presence.
5. Triage real security risk in bounded batches, beginning with Bandit/DeepSource
   dynamic SQL and executable-resolution findings. Distinguish parameterized
   identifier whitelisting and fixed executable contracts from actual injection
   or path-hijack exposure. Add direct regressions for every behavior change;
   do not suppress the backlog broadly.
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
