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
- At this handoff, the Round 21 Binance BBO sidecar (`33804`, `34228`) and the
  Round 27 Stage 1 supervisor (`65092`, `36844`) were alive. The supervisor was
  waiting for fixed slot `stage1-b`; it had no credentials, orders, or trading
  authority. Reverify the processes and canonical state files before acting,
  and never stop or mutate either capture merely because a PID later changes.

## Completed Foundation

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

1. Before resuming model work, verify clean current `main`, capture liveness,
   current-main CI, Ruff, Vulture, Super-Linter, DeepSource, secret scanning,
   Dependabot, code scanning, and branch inventory live. Do not claim zero
   vulnerabilities when a scanner is unavailable, and do not repeat the full
   suite unless a subsequent source change invalidates its passing evidence.
2. Preserve every active real-data capture. Never stop, mutate, relabel, pool, or
   open outcomes from a preregistered capture without explicit operator
   approval and its contract-defined terminal step.
3. When the Stage 1 campaign closes, run only target-blind source, condition,
   role, and population audits. Materialize Round 27, Round 28 BBO, and Round 29
   settlement overlays only after those gates pass.
4. Before any target access, implement and source-bind the Round 29 matched L2
   selection and economic operator specified by
   `round-029-settlement-state-matched-ablation-preregistration-v1.json`.
   Preserve all inherited penalties, scales, costs, delays, and one-use gates.
5. Evaluate only after source, causal split, costs, access ledger, minimum
   sample, and implementation bindings are complete. Failed gates close the
   hypothesis; they never justify weaker costs, adaptive thresholds, or a
   reused holdout.
6. AI stays veto/downsize-only until it demonstrates statistically defensible,
   latency-charged causal uplift over the frozen non-AI decision.
7. After model evidence is credible, finish native Windows usability and
   parser-parity browser/visual QA, then build the manual beta release.

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

Suggested continuation request:

> Read `AGENTS.md`, `docs/CONTINUATION.md`, and `PLANNING.md`; verify the current
> repository, active processes, and GitHub state; then continue the ordered
> work without weakening frozen evidence or safety gates.
